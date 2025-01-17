# xtream AI Challenge

## Ready Player 1? 🚀

Hey there! If you're reading this, you've already aced our first screening. Awesome job! 👏👏👏

Welcome to the next level of your journey towards the [xtream](https://xtreamers.io) AI squad. Here's your cool new assignment.

Take your time – you've got **10 days** to show us your magic, starting from when you get this. No rush, work at your pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### What You Need to Do

Think of this as a real-world project. Fork this repo and treat it as if you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the data and craft your own effective solutions.

🚨 **Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* Your understanding of the data
* The clarity and completeness of your findings
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Keep This in Mind**: This isn't about building the fanciest model: we're more interested in your process and thinking.

---

### Diamonds

**Problem type**: Regression

**Dataset description**: [Diamonds Readme](./datasets/diamonds/README.md)

Meet Don Francesco, the mystery-shrouded, fabulously wealthy owner of a jewelry empire.

He's got an impressive collection of 5000 diamonds and a temperament to match - so let's keep him smiling, shall we?
In our dataset, you'll find all the glittery details of these gems, from size to sparkle, along with their values
appraised by an expert. You can assume that the expert's valuations are in line with the real market value of the stones.

#### Challenge 1

Plot twist! The expert who priced these gems has now vanished.
Francesco needs you to be the new diamond evaluator.
He's looking for a **model that predicts a gem's worth based on its characteristics**.
And, because Francesco's clientele is as demanding as he is, he wants the why behind every price tag.

Create another Jupyter notebook where you develop and evaluate your model.

#### Challenge 2

Good news! Francesco is impressed with the performance of your model.
Now, he's ready to hire a new expert and expand his diamond database.

**Develop an automated pipeline** that trains your model with fresh data,
keeping it as sharp as the diamonds it assesses.

#### Challenge 3

Finally, Francesco wants to bring your brilliance to his business's fingertips.

**Build a REST API** to integrate your model into a web app,
making it a cinch for his team to use.
Keep it developer-friendly – after all, not everyone speaks 'data scientist'!

#### Challenge 4

Your model is doing great, and Francesco wants to make even more money.

The next step is exposing the model to other businesses, but this calls for an upgrade in the training and serving infrastructure.
Using your favorite cloud provider, either AWS, GCP, or Azure, design cloud-based training and serving pipelines.
You should not implement the solution, but you should provide a **detailed explanation** of the architecture and the services you would use, motivating your choices.

So, ready to add some sparkle to this challenge? Let's make these diamonds shine! 🌟💎✨

---

## How to run

### Challenge 1
Create a conda environment with the following command:
```
$ conda create --name xtream --file requirements.txt -c conda-forge
```
The notebook is diamond_price_prediction.ipynb. Just activate the enviroment and start jupyter notebook server with:
```
$ conda activate xtream
$ jupyter-notebook
```

### Challenge 2
I have been creating all my data pipelines with Airflow but as far as I could research xtream likes luigi. So, I decided to implement this challenge with that tool. Inside the xtream enviroment run:
```
$ cd luigi
$ python3 train_pipeline.py SaveMetrics --local-scheduler --input-file ../datasets/diamonds/diamonds.csv
```
I know that local scheduler is on beta but it is a good fit for this challenge. Of course, in production environment I won't use beta features.

### Challenge 3
In the same conda enviroment execute this:
```
$ cd restapi
$ uvicorn app:app --reload
```

The API has three endpoints:
* predict_one: receives a JSON containing the features (x, y, z, color, clarity). All the features are mandatory. Example command:
```
$ curl -X POST http://localhost:8000/predict_one --data '{"x":2, "y":3, "z":4, "color":"H", "clarity":"SI1"}' -H "accept: application/json" -H "Content-Type: application/json"
```
The response is a JSON with an only one key called "price". Example: {"price":5}.
* predict_many: receives a JSON containing a key called "data" and inside it a list, with the features and their IDs. The IDs are mandatory in order to help the web developer with the price-id mapping. All the features are mandatory. Example command:
```
$ curl -X POST http://localhost:8000/predict_many --data '{"data":[{"id": 2, "x":2, "y":3, "z":4, "color":"H", "clarity":"SI1"}, {"id": 1, "x": 1, "y": 1, "z": "50", "color":"F", "clarity": "VS2"}]}' -H "accept: application/json" -H "Content-Type: application/json"
```
The response has a key called "results". Its values is a JSON object containing the IDs as the keys and their respective predictions as values. Example: {"results":{"2":5,"1":6}}.
* predict_many_csv: receives a CSV file containing the features and the IDs. As in predict_many, IDs and features are mandatory. Example command:
```
$ curl -X POST -H "Content-type: multipart/form-data" -F "file=@data.csv;type=text/csv" http://localhost:8000/predict_many_csv
```
The response is the same CSV file but with a new column called "price" with the predictions.

### Challenge 4

I'm going to choose GCP because a I finished some Coursera's MOOCs on that platform.

Architecture:
![Architecture](challenge4.png)

1. Transactional data can be stored as CSVs in a data lake (Google Cloud Storage buckets).
2. Using Google Cloud Dataflow we can run ETLs. The target would be a data warehouse (Google Cloud SQL) which is better than the datalake for exploding data because its scalibility, usability, etc.
3. A luigi instance with all the train pipelines can be installed inside a Docker container (Google Compute Engine). Files (encoder, scaler, model) can be stored in a bucket (Google Cloud Storage). Alternatively, Google AI platform can be used for ML pipeline.
4. Model's API deployment can be done with Google Cloud Run. On the other hand, Google AI platform's prediction service can be used for model deployment. Also, it isn't mandatory to train the model before deploying with Google AI platform, so the previous step remains independent
5. Google Cloud Operations suite can be used for monitoring and logging.
