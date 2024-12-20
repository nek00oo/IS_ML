from lxml import html
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import re
import time
import csv


def save_to_tsv(data, filename):
    all_headers = {}

    for tank in data:
        for key in tank.keys():
            all_headers[key] = None

    all_headers = list(all_headers.keys())

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(all_headers)

        for tank in data:
            row = [tank.get(header, '') for header in all_headers]
            writer.writerow(row)

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
    time.sleep(0.5)
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
    last_highlighted_label = None

    def process_value(base_label, value_text):
        if '/' in value_text:
            values = value_text.split('/')
            for i, val in enumerate(values):
                stats[f"{base_label} part {i + 1}"] = val.strip()
        else:
            stats[base_label] = value_text

    for element in elements:
        stat_lines = element.xpath('.//div[contains(@class, "stat-line")]')

        for stat_line in stat_lines:
            label = stat_line.xpath('.//label/text()')
            value = stat_line.xpath('.//span//text()')

            if label and 'highlight' in stat_line.get('class', ''):
                last_highlighted_label = label[0].strip()

            if label and value:
                label_text = label[0].strip()
                value_text = value[0].strip().replace(',', '')

                if 'repaired' in label_text:
                    continue

                if label_text.startswith('…') and last_highlighted_label:
                    full_label = f"{last_highlighted_label} {label_text.strip('…').strip()}"
                    process_value(full_label, value_text)
                else:
                    if label_text == "Tank cost":
                        class_name = stat_line.xpath('.//span/*/@class')[0]
                        label_text = f"{label_text} ({class_name})"
                    process_value(label_text, value_text)
    return stats

def parse_tank_description(description):
    pattern = r"(?P<tier>Tier\s+\w+)(?P<premium_reward_event>\s+(Premium|Reward|Event))?\s+(?:Steel\s+Hunter\s+)?(?P<nationality>\w+)\s+(?P<type>.*)"

    match = re.match(pattern, description)
    if match:
        tank_info = match.groupdict()
        return {
            'Tier': tank_info['tier'].split()[-1],
            'Premium': tank_info['premium_reward_event'] == ' Premium',
            'Reward': tank_info['premium_reward_event'] == ' Reward',
            'Event': tank_info['premium_reward_event'] == ' Event',
            'Nationality': tank_info['nationality'],
            'Type': tank_info['type']
        }
    else:
        return None

def parse_samples(htm):
    tree = html.fromstring(htm)

    tank_name = tree.xpath('//div[@class="tank"]/h1/text()')[0].strip()
    parts = tree.xpath('//div[@class="tank"]/h1/small/text()')[0].strip()

    category = parse_tank_description(parts)

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

    save_to_tsv(all_tanks_data, 'tanks_data.tsv')