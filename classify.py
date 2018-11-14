from sklearn.externals import joblib

# Load the trained model
classifier = joblib.load("job_classifier.pkl")

# 

dir(classifier)

help(classifier.prob_classify)

print(classifier.labels)