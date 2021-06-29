# ############################ Webscrapping Options & Proxy #############################
import io, os, json, random, re, bs4, requests, itertools
from collections import defaultdict
from scipy import stats
from pprint import pprint
from time import sleep, time
import pandas as pd
import numpy as np
from IPython.core.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from http_request_randomizer.requests.proxy.requestProxy import RequestProxy

# Selenium Driver Options
chrome_options = Options()
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
chrome_options.add_argument('--disable-infobars')
chrome_options.add_argument('--disable-extensions')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument('--profile-directory=Default')
chrome_options.add_argument("--incognito")
chrome_options.add_argument("--disable-plugins-discovery")
chrome_options.add_argument("--proxy-server='direct://'")
chrome_options.add_argument("--proxy-bypass-list=*")
chrome_options.add_argument('--headless')
chrome_options.add_argument('--start-maximized')
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('log-level=3')

file_path = lambda x: os.path.join(os.getcwd(), 'data', x)


# ############################ Creating webdriver object ##################################


def get_driver(desired_capabilities=None):
    " function load chrome driver file based on os"
    try:
        driver = webdriver.Chrome(os.path.join(os.getcwd(), 'lib', 'chromedriver'), options=chrome_options,
                                  desired_capabilities=desired_capabilities)
    except:
        driver = webdriver.Chrome(os.path.join(os.getcwd(), 'lib', 'chromedriver_win'), options=chrome_options,
                                  desired_capabilities=desired_capabilities)
    return driver


# ############################# Adding proxies ################################# #

req_proxy = RequestProxy(log_level=50)  # different number of proxy on each run
proxies = req_proxy.get_proxy_list()  # this will create proxy list


def change_proxy(proxies=proxies):
    "function return new driver object with different proxy"

    while True:
        print(f'changing proxy...{proxies[random.randint(1, len(proxies))].get_address()}')
        PROXY = proxies[random.randint(1, len(proxies))].get_address()
        response = requests.get(HOMEPAGE, PROXY)
        # Throw a warning for non-200 status codes
        if response.status_code != 200:
            print('ERROR')
            warn(f'Request: {request}; Status code: {response.status_code}')
        else:
            print(f'Successful connection, Status code: {response.status_code}')
            desired_capabilities = webdriver.DesiredCapabilities.CHROME['proxy'] = {"httpProxy": PROXY,
                                                                                    "ftpProxy": PROXY,
                                                                                    "sslProxy": PROXY,
                                                                                    "proxyType": "MANUAL"}
            DRIVER = get_driver(desired_capabilities)
            return DRIVER


# ############################# Extracting links ################################# #

def get_links(movie_list, savefile, verbose=False):
    """ get links for the name of movies provide using google search """
    links = []
    links_dict = defaultdict(str)
    DRIVER = change_proxy()
    i = 1

    # getting url for every movie
    print(f'Getting links for...')
    for name in movie_list:
        named = ''.join([n for n in name if n.isalpha() or n == ' ' or n.isdigit()])
        url = f'http://www.google.com/search?q={"+".join(named.lower().split())}+imdb'
        DRIVER.get(url)
        try:
            href_obj = [l.get_property('href') for l in DRIVER.find_elements_by_tag_name('a')]
            link = re.search('https://www.imdb.com/title/tt(\d)*', ' '.join(href_obj)).group()
            links.append(link)
        except:
            print('error')
            links.append('error')

        # optional argument for printing outputs 
        if verbose:
            print(f'{name} - {link}')

        links_dict[name] = link

        # change proxy and bk to file every 20 links before bot trap
        if i % 20 == 0 or i == len(movie_list):
            DRIVER = change_proxy()
            sleep(DELAY)
            backup_info(savefile, links_dict, True)
        i += 1
    return links_dict


# ############################# Extracting movie info ##################################

def get_movie_data(file_data, method='all'):
    """ scarp movies with missing info and return values by method argument"""
    page_html = ''
    i = 0
    DRIVER = get_driver()
    DRIVER.get(HOMEPAGE)
    titles = []
    try:
        file_found = os.path.isfile(file_path(file_data))
        if file_found and file_data.endswith('json'):
            print('Reading JSON File...')
            temp_df = pd.read_json(file_data)
            links = temp_df['Links']
        else:
            links = backup_info(file_data, '')
            print('Reading TXT File...')
            data_dict = {'Name': links.keys(),
                         'Links': links.values(),
                         'Lead actor\ess': pd.Series(np.ones((len(links.keys()))), dtype=str),
                         'Score': pd.Series(np.ones((len(links.keys())))),
                         'Type': pd.Series(np.ones((len(links.keys()))), dtype=str)}
            temp_df = pd.DataFrame(data_dict)
            links = temp_df['Links']
            temp_df['Name'] = temp_df['Name'].apply(lambda x: x.strip('\n'))
        ind = temp_df.index.tolist()
    except:
        print('Reading DataFrame...')
        temp_df = file_data
        links = temp_df['Links'].apply(lambda x: TITLE_LINK + x)
        ind = temp_df[temp_df['Lead actor\\ess'].isnull()].index

    if method == 'all':
        method = ['score', 'actor', 'type', 'year', 'title']
    else:
        mathod = [method]

    for link in links:
        if link not in ['ERROR', '', None]:
            print(f'\nGetting info on:   {link}')
            DRIVER.get(link)
            try:
                wait = WebDriverWait(DRIVER, 10)
                page_html = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'title-overview')))[0].text
            except selenium.common.exceptions.TimeoutException:
                DRIVER = change_proxy()
                sleep(DELAY)
                try:
                    wait = WebDriverWait(DRIVER, 20)
                    page_html = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'title-overview')))[
                        0].text
                except selenium.common.exceptions.TimeoutException:
                    print('Cant find element...')
                    temp_df.at[ind[i], 'Lead actor\ess'] = None
                    temp_df.at[ind[i], 'Score'] = None
                    temp_df.at[ind[i], 'Type'] = None
                    temp_df.at[ind[i], 'Title'] = None
                    continue
            html_list = page_html.split('\n')

            # Getting movie Title
            if 'title' in method:
                try:
                    title = ' '.join(html_list[5].split()[:-1])
                    print("TITLE: ", title)
                    temp_df.at[ind[i], 'Title'] = title
                except ValueError:
                    temp_df.at[ind[i], 'Title'] = None
                titles.append(title)

            # Getting movie score
            if 'score' in method:
                try:
                    score = float(html_list[2][:-3])
                    print("SCORE: ", score)
                    temp_df.at[ind[i], 'Score'] = score
                except ValueError:
                    temp_df.at[ind[i], 'Score'] = None

            # Getting movie release year
            if 'year' in method:
                try:
                    year = re.search('[\(]*(19\d{2}|20\d{2})[\)]*', ''.join(html_list)).group(1)
                    print("YEAR: ", year)
                    temp_df.at[ind[i], 'Year'] = year
                except ValueError:
                    temp_df.at[ind[i], 'Year'] = None

            # Getting movie lead actor/ess 
            if 'actor' in method:
                try:
                    actor = ''.join(re.split('Star[s]*:', page_html)[1].split(',')[0])
                    print("ACTOR: ", actor)
                    temp_df.at[ind[i], 'Lead actor\ess'] = actor
                except (AttributeError, IndexError):
                    try:
                        actor = DRIVER.find_element_by_class_name('plot_summary')
                        actor = actor.find_element_by_tag_name(a)
                        temp_df.at[ind[i], 'Lead actor\ess'] = actor.text
                        print("Actor:", actor.text)
                    except:
                        print('Cant find ACTOR object ')
                        temp_df.at[ind[i], 'Lead actor\ess'] = None

            # Getting movie type
            if 'type' in method:
                try:
                    mtype = re.search('\d*h \d*min \| ([A-z\s]*)', page_html).group(1)
                    print('TYPE: ', mtype)
                    temp_df.at[ind[i], 'Type'] = mtype
                except AttributeError:
                    try:
                        mtype = DRIVER.find_element_by_class_name('title_wrapper')
                        mtype = [t.text for t in mtype.find_elements_by_tag_name('a')]
                        mtype = [j for j in mtype if j in uni_type][0]
                        temp_df.at[ind[i], 'Type'] = mtype
                        print('Type: ', mtype)
                    except:
                        print('Cant find TYPE object ')
                        temp_df.at[ind[i], 'Type'] = None

        # backup file every 20 links
        i += 1
        if i % 20 == 0 and not isinstance(file_data, pd.DataFrame) or i == len(ind):
            backup_info(file_data[:-4] + '_df.json', temp_df)

    return temp_df


# ############################# Backup files ##################################

def backup_info(filename, data=None, overwrite=False):
    "function check if file found on path and load it or replace it by overwrite argument"
    filename = file_path(filename)
    file_found = os.path.isfile(filename)

    # Writing File
    if not file_found or overwrite:
        print('Saving to file...')
        if isinstance(data, pd.DataFrame):
            data.to_json(filename)
        else:
            with open(filename, 'w') as f:
                f.write(json.dumps(data))
        return data
    # Reading File
    else:
        print('File Found...')
        if filename.endswith('txt'):
            with open(filename) as f:
                return json.loads(f.read())
        else:
            return pd.read_json(filename)

        return data


# ############################# Apply Threading ##################################

def apply_thread(func, urls):
    MAX_THREADS = 30
    threads = min(MAX_THREADS, len(urls))
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(func, urls)


# ############################# Finding Nulls ##################################

def missing_values(data):
    """
    :param data:  data set
    gets a data set and plot the amount and percentage of missing values in each feature
    """
    columns_missing_values = (data.count() / len(data)) < 1
    missing_values = data.loc[:, columns_missing_values].isnull().sum().sort_values(ascending=False)
    missing_value_df = pd.concat([missing_values, 100 * round(missing_values / len(data), 3)], axis=1, keys=["#", "%"])
    display(missing_value_df)


# ############################# Plot group distribution ##################################

def display_group_density_plot(df, groupby, on, palette=None, figsize=(16, 5), xlim=None):
    """
    Displays a density plot by group, given a continuous variable, and a group to split the data by
    :param df: DataFrame to display data from
    :param groupby: Column name by which plots would be grouped (Categorical, maximum 10 categories)
    :param on: Column name of the different density plots
    :param palette: Color palette to use for drawing
    :param figsize: Figure size
    :return: matplotlib.axes._subplots.AxesSubplot object
    """

    if not isinstance(df, pd.core.frame.DataFrame):
        raise ValueError('df must be a pandas DataFrame')

    if not groupby:
        raise ValueError('groupby parameter must be provided')

    elif not groupby in df.keys():
        raise ValueError(groupby + ' column does not exist in the given DataFrame')

    if not on:
        raise ValueError('on parameter must be provided')

    elif not on in df.keys():
        raise ValueError(on + ' column does not exist in the given DataFrame')

    if len(set(df[groupby])) > 10:
        groups = df[groupby].value_counts().index[:10]

    else:
        groups = set(df[groupby])

    # Get relevant palette
    if palette:
        palette = palette[:len(groups)]
    else:
        palette = sns.color_palette()[:len(groups)]

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for value, color in zip(groups, palette):
        sns.kdeplot(df.loc[df[groupby] == value][on], shade=True, shade_lowest=True, color=color, label=value)

    ax.set_title(str("Distribution of " + on + " per " + groupby + " group"), fontsize=20)
    ax.set_xlabel(on, fontsize=20)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.tight_layout()
    plt.xlim(xlim)
    plt.show();


# ############################# Plot Categorical distribution ##################################

def plot_cat_dist(df, cols):
    sns.set(font_scale=1.5)
    sns.set_style("darkgrid")
    for i, col in enumerate(cols):
        plt.subplots(figsize=(70, 5))
        plt.subplot(1, len(cols), i + 1)
        g = sns.countplot(df[col], data=df, order=df[col].value_counts().iloc[:15].index, palette="Set2")
        for p in g.patches:
            height = p.get_height()
            g.text(p.get_x() + p.get_width() / 2., height + 0.4, height, ha="center")
        g.set_title(label=col + " distribution", fontsize=20)
        plt.tight_layout()
        plt.show()
