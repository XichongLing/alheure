# À l'heure
**Best AI for Commerce (Stonks) Winner at MAIS Hacks 2021 hosted by McGill Artificial Intelligence Society**   
This application helps the user predict whether a flight will be on time or delayed.

## Try it out
* [Application](https://cs.mcgill.ca/~zjiang27/projects/alheure)
* [Presentation on Devpost](https://devpost.com/software/smart-delay)

## Created by...
* [Junjian Chen](https://github.com/JoeyChen-95)
* [Zhekai Jiang](https://github.com/zhekai-jiang)
* [Xichong Ling](https://github.com/XichongLing)
* [Shichang Zhang](https://github.com/Shichang-Zhang)

## Inspiration
In today's world, the plane is a necessary vehicle for long trips. However, because of factors like air traffic, weather, departure time, etc., flights often delay. We want to develop a web app to help people predict whether their flights will delay so that they can better schedule their trips.

## What it does
We want to predict whether a flight will delay or not, given the day of the week, departure time, airliner, departure airport, and the previous airport.

## How we built it
We used sklearn in Python to train a model of decision tree using almost 500,000 data points in the dataset [2019 Airline Delays w/Weather and Airport Detail](https://www.kaggle.com/threnjen/2019-airline-delays-and-cancellations) on Kaggle to perform prediction on the user's input. The model is stored on a Flask-based backend server deployed on Heroku to which the frontend will make HTTP requests to perform prediction. The result of the prediction will be displayed on the frontend based on JavaScript, HTML, and CSS.

## Challenges we ran into
* Training our model with a huge amount of data
* Preprocessing data
* Building the website 
* Dependency issues when hosting the Flask backend on Heroku

## Accomplishments that we're proud of
* *Working* frontend and backend
* High accuracy
* Delicate frontend
* Having learned Flask

## What we learned
* Better understanding of decision tree
* Use of Flask framework
* Project architecture
* Project management

## What's next for À l'heure
* Allow more input from user (e.g. weather, size of the plane, month, etc.) to improve accuracy
* Connect to other apps to prefill or export data
* More precise prediction of, for example, probability of delay and duration of delay

## Built with...
* Python
* JavaScript
* HTML & CSS
* Flask
* Heroku
