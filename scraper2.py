import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from io import StringIO
from itertools import chain
from typing import Any, Callable, Dict, List, Tuple, Union
from urllib.parse import urljoin

import cloudscraper
from lxml import etree as et

BASE_URL = "https://www.autotrader.co.uk"
'''Autotrader base url'''

BASE_SEARCH_URL = urljoin(BASE_URL, "results-car-search")
'''Autotrader base search url'''

HTML_PARSER = et.HTMLParser()
'''LXML HTML parser'''

SPEC_RE = re.compile(r"""
	(?:(?P<year>\d{4})(?:[ ]\(\d+[ ]\w+\))?)|								# e.g. 2012
	(?:(?P<mileage>[\d,]+)[ ]miles)|										# e.g. 1,234 miles
	(?:(?P<cc>\d+(?:\.\d+)?)L)|												# e.g. 2.4L
	(?:(?P<power>\d+((PS)|(BHP)|(HP)|(KW))))|								# e.g. 230PS
	(?:(?P<body>((Saloon)|(Hatchback)|(Estate)|(Convertible)|(Coupe)|(SUV))))|	# e.g. Saloon
	(?:(?P<trans>((Automatic)|(Manual))))|									# e.g. Automatic
	(?:(?P<fuel>((Petrol.*Hybrid)|(Petrol)|(Diesel)|(Bi[ ]Fuel))))|			# e.g. Petrol
	(?:(?P<owners>\d+)[ ]owners?)|											# e.g. 2 owners
	(?:(?P<ulez>ULEZ))|  													# e.g. ULEX
	(?:(?P<history>(:?[\w ]+History[\w ]*))) 								# e.g. Full dealership history
	""",
	re.VERBOSE | re.IGNORECASE
)
'''This RE captures information from the advert and categorises it using the groupname''' 

POWER_RE = re.compile(r'(?P<power>\d+)\s*(?P<unit>[A-Z]+)', re.IGNORECASE)
'''This RE splits typical power specifications (such as 220BHP) into (200, "BHP")'''

POWER_TO_BHP = {
	'PS': 0.9863,
	'BHP': 1,
	'KW': 1.341,
}
'''A dict with conversion rates to BHP'''

def power_string_to_bhp(power_string: str) -> int:
	''' Converts a typical power string, such as
	'123BHP' or '456PS'
	into 
	123 or 456*0.98... respectively
	power_string: The string to attempt to convert
	returns: The value in BHP, if converted, otherwise 0
	'''
	m = POWER_RE.search(power_string)
	if m:
		gd = m.groupdict()
		power = gd['power']
		unit = str(gd['unit']).upper()
		factor = POWER_TO_BHP.get(unit, 0)
		# note the conversion to float first to prevent e.g. int('123.4') causing an error
		value = int(float(power) * factor)
		return value
	return 0

PRICE_STR_TO_INT: Callable[[str], int] = lambda p: int(re.sub('[£,]', '', p))
'''This lambda converts a price string (e.g. £1,234) into an int (e.g. 1234)'''

CONVERT_TO_STR_AND_STRIP: Callable[[Any], Any] = lambda x: str(x).strip()
'''This lambda takes an objects, converts it to a string, rips off any whitespace, and returns it'''

CONVERTERS: Dict[str, Callable[[Any], Any]] = {
	'year': 		lambda v: int(v),
	'mileage': 		lambda v: int(str(v).replace(',', '')),
	'cc': 			lambda v: float(v),
	'power': 		power_string_to_bhp,
	'body': 		CONVERT_TO_STR_AND_STRIP,
	'trans': 		CONVERT_TO_STR_AND_STRIP,
	'fuel': 		CONVERT_TO_STR_AND_STRIP,
	'owners': 		lambda v: int(v),
	'ulez': 		CONVERT_TO_STR_AND_STRIP, 
	'history': 		CONVERT_TO_STR_AND_STRIP,
}
'''A dictionary of functions that will convert fetched data'''

CarInfo = Dict[str, Union[str, int]]
'''Alias for a typical datatype'''

LOGGER = logging.getLogger(__name__)

'''An alias for the http code 200 (i.e. OK)'''
HTTP_OK = 200

# logging.basicConfig(level='WARNING')



def scrape_listing(car_listing: et.ElementBase) -> CarInfo:
	'''This function takes a car listing LXML object and gathers info from it
	car_listing: An LXML element to grab the data from
	returns: A dict of data about the advert'''

	# the li itself contains some interesting info, fetch that first
	attrs = dict(car_listing.attrib)

	# get the article and contained within the li
	# use it to decide if the entry is an ad
	advert_article = car_listing.find('article')
	if advert_article.attrib.get("data-standout-type", None) == "promoted":
		LOGGER.debug('Ignoring advert for id %s', attrs.get('id', 'unknown'))
		return {} 

	# get the link to the actual page
	advert_path = advert_article.find('a').attrib['href']
	full_advert_path = urljoin(BASE_URL, advert_path)
	truncated_advert_path, _ = str(full_advert_path).split('?', maxsplit=1)

	# get the price
	price_str = car_listing.find('.//div[@class="product-card-pricing__price"]/span').text
	price = PRICE_STR_TO_INT(price_str)

	# get the title
	title = car_listing.find('.//h3[@class="product-card-details__title"]').text
	title = CONVERT_TO_STR_AND_STRIP(title)

	# then gather this info 
	car_key_spec_dict: CarInfo = {
		'title': title,
		'price': price,
		'url': truncated_advert_path
	}

	# iterate over keys specs and add to dict
	for key_spec in car_listing.xpath('.//ul[@class="listing-key-specs"]/li/text()'):

		# try and match, but fail gracefully
		m = SPEC_RE.fullmatch(key_spec)
		if m is None:
			LOGGER.warning('Did not match a spec for "%s"', key_spec)
			return {} 
		
		# if there was a match, convert the raw text, add to dict
		conv = CONVERTERS.get(m.lastgroup, CONVERT_TO_STR_AND_STRIP)
		car_key_spec_dict[m.lastgroup] = conv(m.groupdict()[m.lastgroup])

	# list out what was matched. T
	LOGGER.debug('Matches were: %s', car_key_spec_dict)

	return car_key_spec_dict

def scrape_page(page: int, others: Dict[Any, Any], max_attempts: int=3) -> Tuple[List[CarInfo], int]:
	''' This function pulls the data from a specified page on Autotrader.
	page: The page to query
	others: A dictionary of params for the cloudscraper object
	max_attempts: The number of tries allowed before giving up
	returns: 2 tuple of: a list of scraped adverts and the maximum number of pages in the query
	'''

	params = dict(others)

	page_data: List[CarInfo] = []

	final_page = 1

	# sometimes cloudscraper can return a 404, even for a good search
	# it is unclear from a quick poke around whether there is any way to 
	# decide if retrying is a good idea or not, so just try a few times
	for attempt in range(max_attempts):

		# create the scraper each time to avoid potentially continuous
		# 404s that can occur when re-using the scraper object
		scraper = cloudscraper.create_scraper()

		# issue request and see if all good
		params["page"] = page
		resp = scraper.get(BASE_SEARCH_URL, params=params)
		LOGGER.info('Scrape page %d, attempt %d, status = %d', page, attempt, resp.status_code)

		# if the request was a failure, retry by jumping to the start of the loop
		if resp.status_code != HTTP_OK:
			continue

		# otherwise load the html into lxml for processing
		resp_json = resp.json()
		html = et.parse(StringIO(resp_json["html"]), HTML_PARSER)
		
		# grab the element containing the info about all links on the page
		nav_info_elem = html.find('.//var[@id="fpa-navigation"]')
		if nav_info_elem is not None:
			nav_info_dict = json.loads(nav_info_elem.attrib.get('fpa-navigation', '{}'))
			# TODO: Handle failure to get this number better
			final_page = nav_info_dict.get('totalPages', page)
			LOGGER.debug('Info dict says search has %d pages', final_page)

		car_lis = html.xpath('.//li[@class="search-page__result"]')
		LOGGER.info('Got %d adverts', len(car_lis))

		# iterate over the found cars, grab the data from them
		page_data.extend( scrape_listing(li) for li in car_lis )
		break

	else:
		# didn't hit break
		LOGGER.warning('Retry failed for page %d', page)

	return page_data, final_page


def scrape(others: Dict[str, Union[str, int]]) -> List[CarInfo]:
	'''Responsible for issuing the http request, reading the response, and 
	controlling what gets pulled for the main "search" page of AutoTrader for a given year.
	It is intended to be a helper function 
	'''

	# get the data from the first page, including how many remaining pages there ar
	first_page_data, total_pages = scrape_page(1, others)

	# if that is all the pages, return the data from there only
	if total_pages < 2:
		return first_page_data

	# prepare the args for pool
	page_range = range(2, total_pages + 1)
	primed_scrape_page = partial(scrape_page, others=others)

	# scrape the data from each page and return the lot
	with ThreadPoolExecutor(max_workers=8) as pool:
		page_data = pool.map(primed_scrape_page, page_range)
	
	# gather the returned data for remaining, throw away the last page number, then flatten and return
	page_data_no_final_page = (ld for ld, _ in page_data if ld)
	car_data: List[CarInfo] = list(filter(None, chain(first_page_data, *page_data_no_final_page)))
	return car_data


def get_cars(
	make: str = "Lexus", 
	model: str = "IS 300", 
	postcode: str = "GL503PY", 
	radius: int=1500, 
	min_year: int=2010, 
	max_year: int=2022,
	max_miles: int=None, 
	trim: str=None, 
	fuel=None, 
	min_power=None, 
	max_power=None, 
	colour=None,
	include_writeoff="exclude") -> List[CarInfo]:


	# Set up parameters for query to autotrader.co.uk
	params = {
		"sort": "relevance",
		"postcode": postcode,
		"radius": radius,
		"make": make,
		"model": model,
		"search-results-price-type": "total-price",
		"search-results-year": "select-year",
		"year-from": min_year,
		"year-to": max_year,
	}

	if trim:
		params['aggregatedTrim'] = trim

	if fuel:
		params['fuel-type'] = fuel

	if min_power:
		params['min-engine-power'] = min_power

	if max_power:
		params['max-engine-power'] = max_power

	if colour:
		params['colour'] = colour

	if max_miles:
		params['maximum-mileage'] = max_miles

	if (include_writeoff == "include"):
		params["writeoff-categories"] = "on"
	elif (include_writeoff == "exclude"):
		params["exclude-writeoff-categories"] = "on"
	elif (include_writeoff == "writeoff-only"):
		params["only-writeoff-categories"] = "on"

	return scrape(params)


if __name__ == "__main__":
	from datetime import datetime
	start = datetime.now()
	get_cars()
	end = datetime.now()
	ts = (end - start).total_seconds()
	print(f'Took {ts} s')
