# ADSAnalyzer: Bibliographic metrics from NASA ADS

ADSAnalyzer extracts data from peer-reviewed publication from [NASA
ADS](https://ui.adsabs.harvard.edu/)i using their
[APIs](https://ui.adsabs.harvard.edu/help/api/). Intructions to run the script can be obtained
by typing

./ADSAnalyzer.py -h

For the given author, the script produces:
    1. A histogram of the number of articles per year, and the cumulative number of citations;
    2. The distribution of papers with citations larger than n, from which the H-index can be extrapolated;
    3. A pie chart representing the number of articles per journal;
    4. A word cloud extracted from papers keywords;
    5. The list of the top 10 co-authors.

## NASA ADS API Token

To make the ADSAnalyzer script work, you should create a personalized API token
from NASA ADS, using the instructions provided [at this
page](https://ui.adsabs.harvard.edu/user/settings/token). The token should be
pasted in a text file called **api_token.txt**, placed in the same folder of
the script. 

## Python requirements

The following python packages, which can be installed through pip install, are required to use ADSAnalyzer:

matplotlib, argparse, requests, numpy, collections, wordcloud
