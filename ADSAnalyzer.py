#!/usr/bin/env python3
import matplotlib.pyplot as plt
import argparse
import requests
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from matplotlib.ticker import MaxNLocator


def plot_keyword_cloud(publications, auth):
   # Extract all keywords from the publications
    keywords = []
    for pub in publications:
        if pub.get("keyword_facet"):
            keywords.extend(pub.get("keyword_facet"))

    # Count keyword frequencies
    keyword_counts = Counter(keywords)

    # Generate the word cloud
    wordcloud = WordCloud(
        background_color="white",
        colormap="viridis"
    ).generate_from_frequencies(keyword_counts)

    # Plot the word cloud
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")  # Turn off axes
    plt.title("Word Cloud of Keywords for " + auth, fontsize=16)
    plt.tight_layout()

    # Save and display the word cloud
    plt.savefig("Keywords_word_cloud.png")
    plt.show()
    
    print("Keyword cloud saved in Keywords_word_cloud.png \n")


def plot_num_papers(publications):

   # Extract citation years from the bibcodes in the citation field
    citation_years = []

    for pub in publications:
        citations = pub.get("citation", [])
        for bibcode in citations:
            if len(bibcode) >= 4 and bibcode[:4].isdigit():  # Check for valid year in the bibcode
                citation_years.append(int(bibcode[:4]))

    # Count citations per year
    citation_counts = Counter(citation_years)

    # Create a sorted list of years and compute cumulative citations
    sorted_years = sorted(citation_counts.keys())
    cumulative_citations = []
    cumulative_sum = 0

    for year in sorted_years:
        cumulative_sum += citation_counts[year]
        cumulative_citations.append(cumulative_sum)


    # Extract publication years for refereed papers
    years = [pub.get("year", "N/A") for pub in publications if pub.get("year") != "N/A"]

    # Count papers per year
    year_counts = Counter(years)

    # Generate histogram
    years_sorted = sorted(year_counts.keys())
    counts_sorted = [year_counts[year] for year in years_sorted]

    fig, ax1 = plt.subplots()

    # Plotting
    ax1.bar(list(map(int, years_sorted)), counts_sorted)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Refereed Papers")

    # Create a secondary y-axis for cumulative citations
    ax2 = ax1.twinx()
    ax2.step(sorted_years, cumulative_citations,  where="post", color="Orange")
    ax2.set_ylabel("Cumulative Number of Citations", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Save and show the plot
    plt.savefig("Papers_per_year.png")
    plt.show()

    print("Distribution of paper per year saved in Papers_per_year.png \n")

def plot_citations_distrib(publications):

    # Create bins from 0 to 1000
    number_of_articles_with_n_citations  = np.zeros(10000)
    number_of_citations                  = np.arange(1, 10001, 1)

    for pub in publications:
       num_cit = pub.get("citation_count") 
       number_of_articles_with_n_citations[0:num_cit+1] = number_of_articles_with_n_citations[0:num_cit+1] + 1

    h_ind = 0
    for i in range(0,1000): 
       if number_of_articles_with_n_citations[i] >= i+1: 
            h_ind = i + 1

    # Plot histogram
    plt.bar(number_of_citations, number_of_articles_with_n_citations, width=0.9)
    plt.bar(h_ind, number_of_articles_with_n_citations[h_ind-1], width=0.9, color='orange')

    plt.xlabel("Number of Citations")
    plt.ylabel("Number of Articles")
    plt.tight_layout()

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    xmax = np.where(number_of_articles_with_n_citations != 0)[0][-1] + 1 if np.any(number_of_articles_with_n_citations != 0) else 0
    
    if xmax > 60:
       xmax = 60

    plt.xlim([0, xmax+1]) 

    if number_of_articles_with_n_citations[0] >= 100:
        plt.yscale('log')


    # Save and show the plot
    plt.savefig("Citations_H_index.png")
    plt.show()

    print("Cumulative distribution of citations saved in Citations_H_index.png \n")

def plot_pie_journals(publications):
   # Collect journal data from bibstem
    journal_counts = Counter()

    for pub in publications:
        bibstem = pub.get("bibstem", [])
        if bibstem:
            journal_counts[bibstem[0]] += 1  # Use the first journal in the list

    # Sort the journal counts from most to least relevant (most articles first)
    sorted_journals = sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)

    # Prepare data for pie chart
    labels, sizes = zip(*sorted_journals)  # Unpack the sorted journal names and counts
    total_papers = sum(sizes)
    percentages = [size / total_papers * 100 for size in sizes]

    # Create an explode effect for the slices (increasing the distance for smaller slices)
    # The larger the percentage, the smaller the explosion distance
    nump = len(percentages)
    explode = []
    step = 0.2/nump
    x = 0
    for p in percentages:
        explode.append(x)
        x = x + step 

    # Plot pie chart
    plt.figure()
    wedges, texts = plt.pie(
        percentages,
        labels=labels,
        startangle=0,
        colors=plt.cm.tab20.colors,
        explode=explode,  
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.5, 'linestyle': 'solid'}
    )


    # Rotate labels radially
    for text, wedge in zip(texts, wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1  # Calculate mid-angle of wedge
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))  

        # Adjust the label alignment based on its quadrant
        if x < 0:  # Left side
            text.set_horizontalalignment('right')
        else:      # Right side
            text.set_horizontalalignment('left')

        if angle < 270:
            angle = angle - 180

        # Rotate the label radially
        text.set_rotation(angle)

    plt.axis("equal")  # Equal aspect ratio ensures the pie chart is a circle
    plt.tight_layout()


    # Save and show the chart
    plt.savefig("Journals_pie_chart.png")
    plt.show()

    print("Pie charts of journals saved in Journals_pie_chart.png \n")

def extract_top_coauthors(publications, topn, auth_facet):
   # Collect coauthor data from author_facet
    coauthor_counts = Counter()

    for pub in publications:
        authors = pub.get("author_facet", [])
        for author in authors:
           if auth_facet[:-1] not in author:  # Exclude yourself
                coauthor_counts[author] += 1

    # Get top 5 coauthors
    top_n_coauthors = coauthor_counts.most_common(topn)


    # Define column widths
    col1_width = 6  # For "Rank"
    col2_width = 20  # For "Author"
    col3_width = 18  # For "Number of Papers"
    header = ["Rank", "Author", "Number of Papers"]

    # Print the header
    print(f"List of top {topn} co-authors: \n")
    print(f"{header[0].ljust(col1_width)}{header[1].ljust(col2_width)}{header[2].rjust(col3_width)}")
    print("-" * (col1_width + col2_width + col3_width))  # Separator line

    # Print the rows
    for i, (author, count) in enumerate(top_n_coauthors, start=1):
        print(f"{str(i).ljust(col1_width)}{author.ljust(col2_width)}{str(count).rjust(col3_width)}")


if __name__ == '__main__':
    with open('api_token.txt', 'r') as f:    
        API_TOKEN = f.readline().strip()

    HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api.adsabs.harvard.edu/v1/search/query"

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Print statistic about one author extracting data from NASA ADS service.")

    # Add optional arguments
    parser.add_argument('--name', type=str, help="Author's name")
    parser.add_argument('--surname', type=str, help="Author's surname")

    # Parse the arguments
    args = parser.parse_args()

    # Check if the arguments are provided; prompt the user for missing ones
    if not args.surname:
        args.surname = input("Please enter author surname: ")

    if not args.name:
        args.name = input("Please enter author first name: ")


    surname = args.surname 
    name = args.name 

    auth       = surname + ', ' + name
    auth_facet = surname + ', ' + name[0] + '.'

    # Query only refereed papers 
    params = {
        "q": f'author:"{auth}" OR author:"{auth_facet}"',
        "fl": "title,year,citation_count,author_facet,bibstem,citation,keyword_facet",
        "fq": "property:refereed",  # Filters to include only refereed publications
        "rows": 20000,  # Number of results
    }

    # Fetch data
    response = requests.get(API_URL, headers=HEADERS, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        exit()

    # Take data
    data = response.json()

    # Process publications
    publications = data.get("response", {}).get("docs", [])
    if not publications:
        print("No publications found.")
        exit()
    else:
        print(f"Found {len(publications)} publications for {name} {surname}. \n")


    plot_num_papers(publications)

    plot_citations_distrib(publications)

    plot_pie_journals(publications)

    plot_keyword_cloud(publications, auth)

    extract_top_coauthors(publications, 10, auth_facet)

