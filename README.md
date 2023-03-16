# SNLP Fact Checking
A corpus-driven fact-checking engine, which returns a confidence value between 0 (fact is false) and 1 (fact is true) given a fact

## Table of contents
* Approach/Data preparation
* Process strategy
* Results
* Used libraries
* Executing the Project
* Team information

## Approach/Data preparation
We used the train dataset in order to build the main skeleton of the engine. In order to do this, we retrieved all the information from the given raw sentence and collected related information in dictionary of dictionaries (corresponding to the each fact).
We also analyzed the distribution of categories among the test dataset and generated charts (bar graphs) to illustrate it:<br/>
<img src="https://github.com/LittleOneNoise/SNLP_Fact_Checking/blob/73169367f6a24f5e6cb95896b12bd29851bdeafb/Images/distrib_cat_train.jpeg" width="400">
<img src="https://github.com/LittleOneNoise/SNLP_Fact_Checking/blob/73169367f6a24f5e6cb95896b12bd29851bdeafb/Images/distrib_cat_test.jpeg" width="400"><br/>
As a result we ended up with the following structure for a sentence that includes the main information that is used for the fact checking.<br/><br/>
Since there are **three types** of sentence structure we handled them with different scenarios:<br/><br/>
`1. Sentence: Albert Einstein's honour is Nobel Prize in Physics.`<br/>
> -> Main information list : ['Albert Einstein', 'Nobel Prize in Physics'], category: honour, synonym of category that exists in other facts as a category: award

`2. Sentence: Nobel Prize in Physics is Albert Einstein's award.`
> -> Main information list: ['Nobel Prize in Physics', 'Albert Einstein'], category: award, synonym of category that exists in other facts as a category: honour

`3. Sentence: Stars Nokia, Finland has been Nokiaed Nokia.`
> -> Main information list: ['None', 'None'], category: stars, synonym does not exist

Notice that the information showed above does not represent the complete data that is included for each element in the dictionary. There is other information such as raw_sentence, splitted_sentence, etc that were used for different purposes for building the skeleton.
In order to explain the skeleton in detail, some important notes related with sentence structures must be given.

Each sentence includes 2 different facts that we call specific fact and generic fact. In example 1 and 2 (given above) Albert Einstein is the specific information while Nobel Prize in Physics stands for generic information.
This priority was used when we built the local database. If you pay attention you will see that sentence structures differ with respect to the position of 'category'. We used this knowledge as a key to prioritize the main information in sentences that have a similar structure as the 2nd example. Furthermore sentences that have a similar structure as given in the 1rst example do not need to be modified, since they are already in prioritized structure (specific information, generic information). Additionally, the sentences like the 3rd example were labelled as False directly, since they do not follow the main structures of sentences and they are not grammatically correct, that leads the engine to crash. Result will be as following, when prioritization technique is applied:

`2. Sentence: Nobel Prize in Physics is Albert Einstein's award.`
> -> Main information list: ['Albert Einstein', 'Nobel Prize in Physics'], category: award, synonym of category that exists in other facts as a category: honour

## Process strategy
The engine uses **2 resources as reference**: Firstly it checks the local database which was built by using the train dataset, then it uses the wikipedia in order to check the fact online.<br/>

* Fact analysis in local database

> Engine seek each fact in the local database according to the common words (which we call them categories). If the given category or synonym of that category matches, it will check how many matches there are in the main information list. Then the following procedure will be applied:
-> If number of matches will be 2, and since category match was confirmed, fact from the test dataset will take the label of the fact from the train dataset (i.e., fact was given in both datasets).
-> If number of matches will be 1, and since category match was confirmed, specific fact (e.g., Albert Einstein) will be seek in the main information list of fact from the test dataset. In this case:
----> if it does not exist there, it means generic information is the reason of one match. That will lead us to not rely on that information.
----> if it exists there, we will check the label of the fact from the train dataset. In this case we will rely only on the label of True in order to label our test fact as False. Otherwise we will not rely on this fact matching, since train fact has False label. These non-reliable facts will be sent to the wikipedia fact checking section.

* Fact analysis on Wikipedia

> Wikipedia search uses not only wikipedia library itself, but also beautifulsoup that handles web-scrapping tasks. First we look for corresponding link by using the wikipedia library. As a result, wikipedia library provides list of links that correspond to each main information which are included by sentence. Using this methodology will prevent the engine to crash because of DisambiguationError or PageError. Using these links in implementation of beautifulsoup library we extracts not only tables from the relevant pages, but also texts. Then we implement OR conditioning such as: 
-> if both (table checking and text checking) returns True, the fact will be labelled as True;
-> if one of them returns True the fact will be labelled as True as well (either text or table includes relevant information);
-> if both return False the fact will be labelled as False (neither text nor table includes relevant information).

## Results
We measured the accuracy of our web scrapping on the train datatset and the following result was obtained:<br/>
`True: 622/1234; False: 612/1234; False positive:  2/1234; False negative: 63/1234; Same as original: 1169/1234; Accuracy: 0.9473`

In order to have an idea about the accuracy of the engine on test dataset we performed the following actions:<br/>
* Check whether a data from the test dataset can be labelled by using the local database (was built upon the train dataset) or not. If so, label the relevant data, otherwise keep it as none;
* Collect the data that were labelled in the previous step, as a sampling dataset;
* Use the sampling dataset in order to do web-scrapping for train dataset (i.e., check the fact online and compute accuracy).<br/>
The result that is described below was obtained by following the steps given above:<br/>
`True: 8/145; False: 137/145; False positive:  0/145; False negative: 8/145; Same as original: 137/145; Accuracy: 0.9448`

At the end, we performed labelling of the test dataset by using, not only online fact checking (only web-scrapping, but not accuracy computation since we don't have true values for all data in the test dataset), but also local database matching. After labelling all the facts from the test dataset, we got the following numbers:<br/>
`657/1342 True Labels and 685/1342 False Labels`

## Used libraries
* drive (from google.colab)<br/>

* tqdm (from tqdm.notebook)
* os
* json
* csv
* re
* Counter (from collections)
* spacy
* matplotlib.pyplot

* unicodedata
* wikipedia
* urllib.request
* urlopen (from urllib.request)
* HTTPError (from urllib.error)
* BeautifulSoup (from bs4)

## Executing the Project
In order to execute the given fact_engine.ipynb apply the following steps:<br/>

NOTE: You must be connected to the internet! Additionally, execute the code in Google Colab. We do not guarantee that it can work on Jupyter Notebook or any other IDE for python.

1. Download the zip folder from the Github repository
2. Extract all files to one folder
3. upload the extracted folder to "My Drive" folder in your Google Drive
4. open the fact_engine.ipynb, which is in the folder that you uploaded, by double clicking on it
5. choose Runtime from the menu above and 'Run all'
6. you will be asked to grant access to your drive, you have to choose the account that in which Google Drive you uploaded the folder and allow access (because the system will read and write files in the folder)

NOTE: Because of incompatibility of colab file with GitHub, some lines might not be seen well in the preview. In this case it would be better check the code in Google Colab itself.

## Team information
**Team Name:** ErasmusVog

**Team Members**:<br/>
- NAMAZOV Mahammad
- CONSTANT Nathanaël
- BEGINO Sofi
