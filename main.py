from lxml import html
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

def initialize_browser():
    chrome_driver_path = "C:\\Users\\valer\\Downloads\\chromedriver-win64\\chromedriver.exe"
    chrome_options = Options()

    chrome_options.add_argument("--headless")

    chrome_service = Service(chrome_driver_path)

    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    return driver

def close_browser(driver):
    driver.quit()

def download_html_dynamic(driver, url):
    driver.get(url)
    time.sleep(1)
    page_source = driver.page_source
    return page_source

def get_tank_html(driver, url):
    base_url = 'https://tanks.gg'
    full_url = base_url + url

    tank_html = download_html_dynamic(driver, full_url)
    if tank_html:
        print(f"Страница {full_url} загружена успешно.")
        return tank_html
    else:
        print(f"Ошибка при загрузке страницы {full_url}.")
        return None

def get_tank_list_url_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    tree = html.fromstring(html_content)

    tanks_block = tree.xpath('//div[@class="tanks"]')[0]

    tank_links = tanks_block.xpath('.//a')

    tank_list_url = []
    for tank in tank_links:
        tank_url = tank.get('href')
        tank_list_url.append(tank_url)

    return tank_list_url

def parse_features(elements):
    stats = {}

    for element in elements:
        stat_lines = element.xpath('.//div[contains(@class, "stat-line")]')

        for stat_line in stat_lines:
            label = stat_line.xpath('.//label/text()')
            value = stat_line.xpath('.//span//text()')

            if label and value:
                label_text = label[0].strip()
                value_text = value[0].strip().replace(',', '')

                if '/' in value_text:
                    values = value_text.split('/')
                    for i, val in enumerate(values):
                        stats[f"{label_text} part {i + 1}"] = val.strip()
                else:
                    stats[label_text] = value_text

    return stats

def parse_category(string):
    if len(string) > 5:
        del string[2]
        if len(string) > 6:
            del string[2]
    category = dict()
    category["Tier"] = string[1]
    category["Nationality"] = string[2]
    category["Type"] = string[3] + ' ' + string[4]

    return category

def parse_samples(htm):
    tree = html.fromstring(htm)

    tank_name = tree.xpath('//div[@class="tank"]/h1/text()')[0].strip()
    parts = tree.xpath('//div[@class="tank"]/h1/small/text()')[0].strip().split()

    category = parse_category(parts)

    weaponry = tree.xpath('//div[@class="mb-3 mb-md-0 ps-xxl-0 col-xxl-auto"]')
    mobility = tree.xpath('//div[@class="col-xxl-auto"]')
    everything = tree.xpath('//div[@class="position-relative col-xxl-auto"]')

    weaponry_features = parse_features(weaponry)
    mobility_features = parse_features(mobility)
    everything_features = parse_features(everything)

    all_features = dict()
    all_features['Name'] = tank_name
    all_features.update(category)
    all_features.update(weaponry_features)
    all_features.update(mobility_features)
    all_features.update(everything_features)

    for key, value in all_features.items():
        print(f"{key}: {value}")

    return all_features

if __name__ == '__main__':
    tanks_data = get_tank_list_url_from_html("tank_list_dynamic.html")
    all_tanks_data = []
    web_driver = initialize_browser()

    for tank_url in tanks_data:
        print(f"Обрабатываю танк {tank_url}...")
        tank_html = get_tank_html(web_driver, tank_url)

        if tank_html:
            tank_info = parse_samples(tank_html)
            all_tanks_data.append(tank_info)
            print(f"Данные о танке {tank_info['Name']} успешно добавлены.\n")

    close_browser(web_driver)