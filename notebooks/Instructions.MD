Step 1:
Prepare the model deployment for production: Update your code to meet common production coding standards and best practices. These include modularization, code quality, proper unit testing, and comments/documentation. Inlcude a data prep function, a function that saves the var_reduced variable as a .yml file, and a function that loads the .yml into a pandas dataframe in addition to the other files. The final GLM equation should not be modified.

Step 2:
Wrap the model code inside an API using FastAPI: The model must be made callable via API call (port 1313). The call will pass 1 to N rows of data in JSON format, and expects a N responses each with a predicted class and probability belonging to the predicted class. 

Here is an example curl call to your API:

curl --request POST --url http://localhost:8080/predict --header 'content-type: application/json' --data '{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"}'

or a batch curl call:

curl --request POST --url http://localhost:8080/predict --header 'content-type: application/json' --data '[{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"},{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.70305872366547", "x3": "-0.522021597308617", ...,"x99": "2.55535888"}]'

Each of the 10,000 rows in the test dataset will be passed through an API call. The call could be a single batch call w/ all 10,000 rows, or 10,000 individual calls. API should be able to handle either case with minimal impact to performance. Reminder: The predictions returned by the API should be the class probability (i.e. decimal value) for belonging to the positive class, the variables used in the model, and the predicted class (defined by the business partner). The results should be in a JSON format.

Step 3:
Wrap your API in a Docker image: Create a Dockerfile that builds your API into an image. Write a shell script titled run_api.sh that runs your image using traditional docker run commands.