from functools import partial
from typing import Any, Callable, Dict, List, Union
import cloudscraper
import logging
from concurrent.futures import ProcessPoolExecutor
from time import sleep
from lxml import etree as et
from io import StringIO
from urllib.parse import urljoin
import re
import json


# Compile a re that will name special fields 
SPEC_RE = re.compile(r"""
	(?:(?P<reg>\d{4})[ ]\(\d+[ ]\w+\))|							# e.g. 2012
	(?:(?P<mileage>[\d,]+)[ ]miles)|							# e.g. 1,234 miles
	(?:(?P<cc>\d+(?:\.\d)?)L)|									# e.g. 2.4L
	(?:(?P<power>\d+)(?P<power_unit>PS|B?HP|KW))|					# e.g. 230PS
	(?:(?P<body>Saloon))|										# e.g. Saloon
	(?:(?P<trans>Automatic|Manual))|							# e.g. Automatic
	(?:(?P<fuel>Petrol[ ]Hybrid))|								# e.g. Petrol
	(?:(?P<owners>\d+)[ ]owners)|								# e.g. 2 owners
	(?:(?P<ulez>ULEZ))|  										# e.g. ULEX
	(?:(?P<history>(:?Full[ ]Dealership[ ]History))) 			# e.g. Full dealership history
	""",
	re.VERBOSE | re.IGNORECASE
)

PRICE_STR_TO_INT: Callable[[str], int] = lambda p: int(re.sub('[$£,]', '', p))

PS_TO_BHP: Callable[[float], float] = lambda ps: ps * 0.9863

NO_CONVERSION: Callable[[Any], Any] = lambda x: x
# A dictionary of functions that will convert fetched 
CONVERTERS: Dict[str, Callable[[Any], Any]] = {
	'reg': 			lambda v: int(v),
	'mileage': 		lambda v: int(str(v).replace(',', '')),
	'cc': 			lambda v: float(v),
	'power': 		lambda v: float(v),
	'body': 		NO_CONVERSION,
	'trans': 		NO_CONVERSION,
	'fuel': 		NO_CONVERSION,
	'owners': 		lambda v: int(v),
	'ulez': 		NO_CONVERSION, 
	'history': 		NO_CONVERSION,
	'power_unit': 	NO_CONVERSION,
}


LOGGER = logging.getLogger(__name__)
STATUS_CODE_SUCCESS = 200

MAX_ATTEMPTS = 3

logging.basicConfig(level='DEBUG')


def scrape(year: int, others: Dict[str, Any]):
	LOGGER.info('year = %d', year)

	# root url
	base_url = "https://www.autotrader.co.uk"
	base_search_url = urljoin(base_url, "results-car-search")
	
	parser = et.HTMLParser()

	car_data: List[Dict[Any, Any]] = []

	# populate page specific vars and scrape
	params = dict(others)
	params["year-from"] = year
	params["year-to"] = year
	params["page"] = page = 1

	# the page on which the scrape will end
	# it will be set during the script to a dict fetched for the page
	final_page = 1

	# sometimes cloudscraper can return a 404, even for a good search
	# it is unclear for a quick poke around whether there is any way to 
	# decide if retrying is a good idea or not, so just try a few times
	for attempt in range(MAX_ATTEMPTS):

		# if not the first attempt, then sleep
		sleep( attempt**2 )

		# create the scraper each time to avoid potentially continuous
		# 404s that can occur when re-using the scraper object
		scraper = cloudscraper.create_scraper()

		# issue request and see if all good
		resp = scraper.get(base_search_url, params=params)
		LOGGER.debug('Attempt %d, status = %d', attempt, resp.status_code)

		# if the request was a success, attempt to grab the info per car
		if resp.status_code == STATUS_CODE_SUCCESS:
			resp_json = resp.json()
			html = et.parse(StringIO(resp_json["html"]), parser)

			
			# grab the element containing the info about all links on the page
			nav_info_elem = html.find('.//var[@id="fpa-navigation"]')
			nav_info_dict = json.loads(nav_info_elem.attrib.get('fpa-navigation', '{}'))
			# TODO: Handle failure to get this number better
			final_page = nav_info_dict.get('totalPages', page)
			LOGGER.info('Info dict says search has %d pages', final_page)

			# # pull out pertinent info
			# current_page = nav_info_dict['currentPage']
			# total_pages = nav_info_dict['totalPages']
			# car_url_data: List[Dict[str, str]] = nav_info_dict['fpaNavigation'][0]['shortAdvertForFPANavigation']

			# LOGGER.info('Got %d listings on page %d/%d', len(car_url_data), current_page, total_pages)

			# # fetch the data for each of the cars on the page
			# for car_url_datum in car_url_data:

			# 	id = car_url_datum['id']
			# 	url = car_url_datum['fpaUrl']  # note this has unicode chars, so  
			# 	full_url = urljoin("https://www.autotrader.co.uk", url)
			# 	LOGGER.info('Finding info for id %s at %s', id, full_url)

			# 	car_data: List[str] = []

			# 	# scrape car info
			# 	for attempt2 in range(MAX_ATTEMPTS):

			# 		resp2 = scraper.get(full_url)



			# 		if resp2.status_code == STATUS_CODE_SUCCESS:

			# 			resp2_content_str = resp2.content.decode('utf-8')
			# 			html2 = et.parse(StringIO(resp2_content_str), parser)
						
			# 			car_info_elems = html2.xpath('.//ul[@data-gui="key-specs-section"]/li')

						
			# 			car_data.append( x.text for x in car_info_elems )


			car_lis = html.xpath('.//li[@class="search-page__result"]')
			LOGGER.info('Got %d adverts', len(car_lis))

			# iterate over the found cars, grab the data from them
			for car_li in car_lis:

				# get the article and contained within the li
				# use it to decide if the entry is an ad
				advert_article = car_li.find('article')
				if advert_article.attrib.get("data-standout-type", None) == "promoted":
					LOGGER.debug('Ignoring advert for id %s', attrs.get('id', 1234))
					continue

				# get the link to the actual page
				advert_a = advert_article.find('a')
				advert_path = advert_a.attrib['href']
				full_advert_path = urljoin(base_url, advert_path)
				LOGGER.debug('Advert path is: %s', full_advert_path)

				# the li itself contains some interesting info, fetch that first
				attrs = dict(car_li.attrib)

				# get the price
				price_str = car_li.find('.//div[@class="product-card-pricing__price"]/span').text
				price = PRICE_STR_TO_INT(price_str)
				LOGGER.debug('Price is %d', price)

				# get the title
				title = car_li.find('.//h3[@class="product-card-details__title"]').text
				title = str(title).strip()
				LOGGER.debug('Title is %s', title)

				# fetch key specs for the advert card
				key_specs = car_li.xpath('.//ul[@class="listing-key-specs"]/li/text()')

				# then gather them into a dict, converting the types if sensible
				car_key_spec_dict: Dict[str, Union[str, int]] = {
					'title': title,
					'price': price
				}
				for key_spec in key_specs:

					# try and match, but fail gracefully
					m = SPEC_RE.fullmatch(key_spec)
					if m is None:
						LOGGER.warning('Did not match a spec for "%s"', key_spec)
						continue 
					
					# if there was a match, grab the converter
					LOGGER.debug('%s matches %s', key_spec, m.lastgroup)
					conv = CONVERTERS.get(m.lastgroup, NO_CONVERSION)
					car_key_spec_dict[m.lastgroup] = conv(m.groupdict()[m.lastgroup])

				LOGGER.debug('Final dict: %s', car_key_spec_dict)


			break
		

	return year


def get_cars(
	make: str = "Lexus", 
	model="IS 300", 
	postcode="GL503PY", 
	radius=1500, 
	min_year=2016, 
	max_year=2016,
	max_miles=None, 
	trim=None, 
	fuel=None, 
	min_power=None, 
	max_power=None, 
	colour=None,
	include_writeoff="exclude"):



	# Basic variables

	results = []
	n_this_year_results = 0

	keywords = {}
	keywords["mileage"] = ["miles"]
	keywords["BHP"] = ["BHP"]
	keywords["transmission"] = ["Automatic", "Manual"]
	keywords["fuel"] = ["Petrol", "Diesel", "Electric", "Hybrid – Diesel/Electric Plug-in", "Hybrid – Petrol/Electric", "Hybrid – Petrol/Electric Plug-in"]
	keywords["owners"] = ["owners"]
	keywords["body"] = ["Coupe", "Convertible", "Estate", "Hatchback", "MPV", "Pickup", "SUV", "Saloon"]
	keywords["ULEZ"] = ["ULEZ"]
	keywords["year"] = [" reg)"]
	keywords["engine"] = ["engine"]

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

	
		
	# year = min_year
	# page = 1
	# attempt = 1

	n_years = (max_year - min_year) + 1

	partial_scrape = partial(scrape, others=params)
	with ProcessPoolExecutor(max_workers=4) as pool:
		res = pool.map(partial_scrape, range(min_year, max_year + 1))

	# for r in res:
	# 	LOGGER.debug(r)

	# try:

	# 	# for year in range(min_year, max_year+1):
	# 	while year <= max_year:

	# 		# populate page specific vars and scrape
	# 		params["year-from"] = year
	# 		params["year-to"] = year
	# 		params["page"] = page
	# 		r = scraper.get(url, params=params)
	# 		LOGGER.debug(f"Year: {year}, Page: {page}, Response: {r}")

	# 		# # try the scrape up to max_attempts_per_page
	# 		# for _ in range(max_attempts_per_page):
	# 		# 	r = scraper.get(url, params=params)
	# 		# 	if r.status_code == STATUS_CODE_SUCCESS:
	# 		# 		break
	# 		# 	sleep(0.5)  # sleep for some random amount of time
	# 		# else:
	# 		# 	page += 1
	# 		# 	continue

	# 		try:

	# 			if r.status_code != STATUS_CODE_SUCCESS: # if not successful (e.g. due to bot protection), log as an attempt
	# 				print('Got bad code %d, retrying', r.status_code)
	# 				attempt = attempt + 1
	# 				if attempt <= max_attempts_per_page:
	# 					if verbose:
	# 						print("Exception. Starting attempt #", attempt, "and keeping at page #", page)
	# 				else:
	# 					page = page + 1
	# 					attempt = 1
	# 					if verbose:
	# 						print("Exception. All attempts exhausted for this page. Skipping to next page #", page)
	# 				return results

	# 			else:

	# 				j = r.json()
	# 				s = BeautifulSoup(j["html"], features="html.parser")

	# 				articles = s.find_all("article", attrs={"data-standout-type":""})

	# 				# if no results or reached end of results...
	# 				if len(articles) == 0 or r.url[r.url.find("page=")+5:] != str(page):
	# 					if verbose:
	# 						print("Found total", n_this_year_results, "results for year", year, "across", page-1, "pages")
	# 						if year+1 <= max_year:
	# 							print("Moving on to year", year + 1)
	# 							print("---------------------------------")

	# 					# Increment year and reset relevant variables
	# 					year = year + 1
	# 					page = 1
	# 					attempt = 1
	# 					n_this_year_results = 0
	# 				else:
	# 					for article in articles:
	# 						car = {}
	# 						car["name"] = article.find("h3", {"class": "product-card-details__title"}).text.strip()
	# 						car["detail"] = article.find("p", {"class": "product-card-details__subtitle"}).text.strip()
	# 						car["link"] = "https://www.autotrader.co.uk" + article.find("a", {"class": "tracking-standard-link"})["href"][: article.find("a", {"class": "tracking-standard-link"})["href"].find("?")]
	# 						car["price"] = article.find("div", {"class": "product-card-pricing__price"}).text.strip()

	# 						# find the location
	# 						seller_info = article.find_all("li", {"class": "product-card-seller-info__spec-item atc-type-picanto"})
	# 						if seller_info:
	# 							guess_loc = seller_info[len(seller_info)-1].find("span", {"class": "product-card-seller-info__spec-item-copy"}).text.strip()
	# 							try:
	# 								float(guess_loc)
	# 								pass
	# 							except ValueError:
	# 								car["location"] = guess_loc

	# 						key_specs_bs_list = article.find("ul", {"class": "listing-key-specs"}).find_all("li")
							
	# 						for key_spec_bs_li in key_specs_bs_list:

	# 							key_spec_bs = key_spec_bs_li.text

	# 							if any(keyword in key_spec_bs for keyword in keywords["mileage"]):
	# 								car["mileage"] = int(key_spec_bs[:key_spec_bs.find(" miles")].replace(",",""))
	# 							elif any(keyword in key_spec_bs for keyword in keywords["BHP"]):
	# 								car["BHP"] = int(key_spec_bs[:key_spec_bs.find("BHP")])
	# 							elif any(keyword in key_spec_bs for keyword in keywords["transmission"]):
	# 								car["transmission"] = key_spec_bs
	# 							elif any(keyword in key_spec_bs for keyword in keywords["fuel"]):
	# 								car["fuel"] = key_spec_bs
	# 							elif any(keyword in key_spec_bs for keyword in keywords["owners"]):
	# 								car["owners"] = int(key_spec_bs[:key_spec_bs.find(" owners")])
	# 							elif any(keyword in key_spec_bs for keyword in keywords["body"]):
	# 								car["body"] = key_spec_bs
	# 							elif any(keyword in key_spec_bs for keyword in keywords["ULEZ"]):
	# 								car["ULEZ"] = key_spec_bs
	# 							elif any(keyword in key_spec_bs for keyword in keywords["year"]):
	# 								car["year"] = key_spec_bs
	# 							elif key_spec_bs[1] == "." and key_spec_bs[3] == "L":
	# 								car["engine"] = key_spec_bs

	# 						results.append(car)
	# 						n_this_year_results = n_this_year_results + 1

	# 					page = page + 1
	# 					attempt = 1

	# 					if verbose:
	# 						print("Car count: ", len(results))
	# 						print("---------------------------------")

	# 		except KeyboardInterrupt:
	# 			break

	# 		except:
	# 			traceback.print_exc()
	# 			attempt = attempt + 1
	# 			if attempt <= max_attempts_per_page:
	# 				if verbose:
	# 					print("Exception. Starting attempt #", attempt, "and keeping at page #", page)
	# 			else:
	# 				page = page + 1
	# 				attempt = 1
	# 				if verbose:
	# 					print("Exception. All attempts exhausted for this page. Skipping to next page #", page)

	# except KeyboardInterrupt:
	# 	pass

	# return results

### Output functions ###

# def save_csv(results = [], filename = "scraper_output.csv"):
# 	csv_columns = ["name", "link", "price", "mileage", "BHP", "transmission", "fuel", "owners", "body", "ULEZ", "engine", "year"]

# 	with open(filename, "w", newline='') as f:
# 		writer = csv.DictWriter(f, fieldnames=csv_columns)
# 		writer.writeheader()
# 		for data in results:
# 			writer.writerow(data)

# def save_json(results = [], filename = "scraper_output.json"):
# 	with open(filename, 'w') as f:
# 		json.dump(results, f, sort_keys=True, indent=4, separators=(',', ': '))

if __name__ == "__main__":
	get_cars()