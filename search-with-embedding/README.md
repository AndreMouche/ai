# Quickstart 

This is an example search from doc app used in the OpenAI API accordding the [quickstart tutorial](https://beta.openai.com/docs/quickstart) from openai. It uses the [Flask](https://flask.palletsprojects.com/en/2.0.x/) web framework. Check out the tutorial or follow the instructions below to get set up.

## Setup

1. If you don’t have Python installed, [install it from here](https://www.python.org/downloads/)

2. Navigate into the project directory

   ```bash
   $ cd $PWD
   ```

3. start redis
   ```bash
   $ cd redis
   $ bash run.sh
   ```

4. Create a new virtual environment

   ```bash
   $ python -m venv venv
   $ . venv/bin/activate
   ```

5. Install the requirements

   ```bash
   $ pip install -r requirements.txt
   ```

6. Make a copy of the example environment variables file

   ```bash
   $ cp .env.example .env
   ```

7. Add your [API key](https://beta.openai.com/account/api-keys) to the newly created `.env` file or export the api key 
   ```bash
   $ export OPENAI_API_KEY=""
   ```

8. get the embedding for each doc and save into redis
    ```bash
   $ python index_doc_cn.py
   ```
   
9. Run the app

   ```bash
   $ flask run
   ```

You should now be able to access the app at [http://localhost:5000](http://localhost:5000)! For the full context behind this example app, check out the [tutorial](https://beta.openai.com/docs/quickstart).
