import logging
from functools import partial
from typing import Any, Callable, Dict, List, Union

import cloudscraper

from concurrent.futures import ThreadPoolExecutor
import json
import re
from io import StringIO
from itertools import chain
from urllib.parse import urljoin

from lxml import etree as et

# root urls
BASE_URL = "https://www.autotrader.co.uk"
BASE_SEARCH_URL = urljoin(BASE_URL, "results-car-search")

'''This RE captures information from the advert and categorises it using the groupname''' 
SPEC_RE = re.compile(r"""
	(?:(?P<reg>\d{4})(?:[ ]\(\d+[ ]\w+\))?)|								# e.g. 2012
	(?:(?P<mileage>[\d,]+)[ ]miles)|										# e.g. 1,234 miles
	(?:(?P<cc>\d+(?:\.\d+)?)L)|												# e.g. 2.4L
	(?:(?P<power>\d+((PS)|(BHP)|(HP)|(KW))))|								# e.g. 230PS
	(?:(?P<body>((Saloon)|(Hatchback)|(Estate)|(Convertible))))|			# e.g. Saloon
	(?:(?P<trans>((Automatic)|(Manual))))|									# e.g. Automatic
	(?:(?P<fuel>((Petrol.*Hybrid)|(Petrol)|(Diesel)|(Bi[ ]Fuel))))|			# e.g. Petrol
	(?:(?P<owners>\d+)[ ]owners?)|											# e.g. 2 owners
	(?:(?P<ulez>ULEZ))|  													# e.g. ULEX
	(?:(?P<history>(:?[\w ]+History[\w ]*))) 								# e.g. Full dealership history
	""",
	re.VERBOSE | re.IGNORECASE
)

'''This RE splits typical power specifications (such as 220BHP) into (200, "BHP")'''
POWER_RE = re.compile(r'(?P<power>\d+)\s*(?P<unit>[A-Z]+)', re.IGNORECASE)

'''A dict with conversion rates to BHP'''
POWER_TO_BHP = {
	'PS': 0.9863,
	'BHP': 1,
	'KW': 1.341,
}

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

'''This lambda converts a price string (e.g. £1,234) into an int (e.g. 1234)'''
PRICE_STR_TO_INT: Callable[[str], int] = lambda p: int(re.sub('[£,]', '', p))

'''This lambda takes an objects, converts it to a string, rips off any whitespace, and returns it'''
CONVERT_TO_STR_AND_STRIP: Callable[[Any], Any] = lambda x: str(x).strip()

'''A dictionary of functions that will convert fetched data'''
CONVERTERS: Dict[str, Callable[[Any], Any]] = {
	'reg': 			lambda v: int(v),
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


LOGGER = logging.getLogger(__name__)

'''An alias for the http code 200 (i.e. OK)'''
HTTP_OK = 200

'''The number of attempts to scrape a given page'''
MAX_ATTEMPTS = 3

logging.basicConfig(level='INFO')

def scrape_listing(car_listing: et.ElementBase) -> Dict[str, Union[int, str]]:
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
	LOGGER.debug('Advert path is: %s', truncated_advert_path)

	# get the price
	price_str = car_listing.find('.//div[@class="product-card-pricing__price"]/span').text
	price = PRICE_STR_TO_INT(price_str)
	LOGGER.debug('Price is %d', price)

	# get the title
	title = car_listing.find('.//h3[@class="product-card-details__title"]').text
	title = CONVERT_TO_STR_AND_STRIP(title)
	LOGGER.debug('Title is %s', title)

	# then gather this info 
	car_key_spec_dict: Dict[str, Union[str, int]] = {
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
		LOGGER.debug('%s matches %s', key_spec, m.lastgroup)
		conv = CONVERTERS.get(m.lastgroup, CONVERT_TO_STR_AND_STRIP)
		car_key_spec_dict[m.lastgroup] = conv(m.groupdict()[m.lastgroup])

	return car_key_spec_dict

def scrape_page():

	pass


def scrape(year: int, others: Dict[str, Any]):
	'''Responsible for issuing the http request, reading the response, and 
	controlling what gets pulled for the main "search" page of AutoTrader for a given year.
	It is intended to be a helper function 
	year: The year  '''

	parser = et.HTMLParser()

	car_data: List[Dict[Any, Any]] = []

	# populate year vars here
	# then the page # in the loop
	params = dict(others)
	params["year-from"] = year
	params["year-to"] = year

	# the page on which the scrape will end
	# it will be set during the script to a dict fetched for the page
	page = 1
	final_page = 1

	while page <= final_page:

		# sometimes cloudscraper can return a 404, even for a good search
		# it is unclear from a quick poke around whether there is any way to 
		# decide if retrying is a good idea or not, so just try a few times
		for attempt in range(MAX_ATTEMPTS):

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
			html = et.parse(StringIO(resp_json["html"]), parser)
			
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
			car_data.extend( scrape_listing(li) for li in car_lis )
			break

		else:
			# didn't hit break
			LOGGER.warning('Retry failed for page %d', page)
		
		# increment the page for the while loop
		page += 1

	return car_data


def get_cars(
	make: str = "Ariel", 
	model: str = "Atom", 
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
	include_writeoff="exclude"):


	# Set up parameters for query to autotrader.co.uk
	params = {
		"sort": "relevance",
		"postcode": postcode,
		"radius": radius,
		"make": make,
		"model": model,
		"search-results-price-type": "total-price",
		"search-results-year": "select-year",
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


	n_years = (max_year - min_year) + 1

	partial_scrape = partial(scrape, others=params)
	with ThreadPoolExecutor(max_workers=4) as pool:
		res = pool.map(partial_scrape, range(min_year, max_year + 1))

	combined_res = chain(*res)

	for r in combined_res:
		LOGGER.info(r)


if __name__ == "__main__":
	from datetime import datetime
	start = datetime.now()
	get_cars()
	end = datetime.now()
	ts = (end - start).total_seconds()
	print(f'Took {ts} s')
