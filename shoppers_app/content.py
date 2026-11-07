"""Static teaching text used by the app (feature meanings, model explanations, metric meanings)."""

FEATURE_INFO = [
    (
        "Administrative",
        "Count",
        "Number of account / admin pages (login, account settings) the visitor opened.",
        "Shows how much the visitor interacted with their account.",
    ),
    (
        "Administrative_Duration",
        "Seconds",
        "Total time spent on administrative pages.",
        "Long admin time can mean a registered user managing an order or checkout details.",
    ),
    (
        "Informational",
        "Count",
        "Number of informational pages (about us, contact, help) visited.",
        "Reading help pages can mean the visitor is hesitating or comparing.",
    ),
    (
        "Informational_Duration",
        "Seconds",
        "Total time spent on informational pages.",
        "Same idea, measured in time.",
    ),
    (
        "ProductRelated",
        "Count",
        "Number of product pages visited.",
        "More product pages usually means more browsing intent.",
    ),
    (
        "ProductRelated_Duration",
        "Seconds",
        "Total time spent on product pages.",
        "A strong engagement signal, but very skewed (a few very long sessions).",
    ),
    (
        "BounceRates",
        "Ratio 0-1",
        "Share of visits to the pages in this session that left after one page.",
        "High bounce rate means visitors leave quickly, so purchase is unlikely.",
    ),
    (
        "ExitRates",
        "Ratio 0-1",
        "Average share of page views that were the last in the session.",
        "High exit rate means the visitor tends to leave from the pages they see.",
    ),
    (
        "PageValues",
        "Value",
        "Average value of the pages visited before the visitor completed a transaction.",
        "Usually the strongest signal. It is computed using transaction outcomes, so it partly knows the answer.",
    ),
    (
        "SpecialDay",
        "0 to 1",
        "Closeness of the visit to a special day (e.g. Valentine's Day, Mother's Day).",
        "Shoppers may buy more near these days, but they often also compare more.",
    ),
    (
        "Month",
        "Category",
        "Month of the visit.",
        "Shows seasonality (e.g. Nov and May are busy).",
    ),
    (
        "OperatingSystems",
        "Category code",
        "Operating system of the visitor (coded as a number).",
        "Can reflect device type or customer group.",
    ),
    (
        "Browser",
        "Category code",
        "Browser of the visitor (coded as a number).",
        "Can reflect device type.",
    ),
    (
        "Region",
        "Category code",
        "Geographic region of the visitor (coded as a number).",
        "Buying behaviour can differ by region.",
    ),
    (
        "TrafficType",
        "Category code",
        "How the visitor arrived (ads, search, direct, ... coded as a number).",
        "Different channels bring visitors with different intent.",
    ),
    (
        "VisitorType",
        "Category",
        "New_Visitor, Returning_Visitor or Other.",
        "Returning visitors know the shop, which changes buying behaviour.",
    ),
    (
        "Weekend",
        "Boolean",
        "True if the visit was on a weekend.",
        "Weekend shoppers may behave differently.",
    ),
    (
        "Revenue",
        "Boolean (target)",
        "True if the visit ended with a purchase.",
        "This is what we want to predict.",
    ),
]

FEATURE_ENG = {
    "TotalPages": dict(
        what="All pages visited in the session (admin + informational + product).",
        formula="Administrative + Informational + ProductRelated",
        why="The three page counts describe one thing, how much the visitor browsed. One total is easier for a model to use.",
        cols=["Administrative", "Informational", "ProductRelated"],
    ),
    "TotalDuration": dict(
        what="Total time on the site, in seconds.",
        formula="Administrative_Duration + Informational_Duration + ProductRelated_Duration",
        why="Gives one overall engagement measure instead of three separate times.",
        cols=[
            "Administrative_Duration",
            "Informational_Duration",
            "ProductRelated_Duration",
        ],
    ),
    "AvgTimePerPage": dict(
        what="Average seconds spent per page.",
        formula="TotalDuration / TotalPages  (0 when TotalPages is 0)",
        why="Separates careful readers (few pages, long time) from quick skimmers (many pages, short time).",
        cols=["TotalDuration", "TotalPages"],
    ),
    "HasPageValue": dict(
        what="1 if the visitor saw pages with a positive PageValue, else 0.",
        formula="1 if PageValues > 0 else 0",
        why="Most visits have PageValues = 0. A simple yes/no flag makes that split easy for the models.",
        cols=["PageValues"],
    ),
    "IsReturning": dict(
        what="1 if the visitor has visited before, else 0.",
        formula="1 if VisitorType == 'Returning_Visitor' else 0",
        why="Turns the visitor type into one clear yes/no signal.",
        cols=["VisitorType"],
    ),
    "IsHolidaySeason": dict(
        what="1 if the visit was in November or December, else 0.",
        formula="1 if Month in ['Nov', 'Dec'] else 0",
        why="Shopping peaks around the holidays, so this groups two months into one seasonal signal.",
        cols=["Month"],
    ),
    "log_ProductRelated_Duration": dict(
        what="Natural log of (1 + time on product pages).",
        formula="log(1 + ProductRelated_Duration)",
        why="The raw column is very skewed. The log squeezes very large values so they do not dominate the models.",
        cols=["ProductRelated_Duration"],
    ),
}

MODEL_INFO = {
    "Logistic Regression": dict(
        principle="Learns a weight for every feature and combines them into a probability of purchase. "
        "The decision boundary is a straight line (a flat plane in many dimensions).",
        approach="Each browsing signal (for example PageValues or ExitRates) pushes the purchase probability up or down. "
        "A visit is called a buyer when the probability is above 0.5.",
        settings="StandardScaler, then LogisticRegression(max_iter=1000)",
        scaled=True,
        strengths=[
            "Fast and stable",
            "Weights can be read and explained",
            "Gives well-behaved probabilities",
        ],
        limits=[
            "Only learns straight-line boundaries",
            "Cannot capture feature interactions without extra work",
        ],
        choose="You need a simple, explainable baseline and trustworthy probabilities.",
        avoid="The real relationship is strongly non-linear or full of interactions.",
    ),
    "Decision Tree": dict(
        principle="Asks a chain of yes/no questions about the features (for example 'is PageValues > 0?') "
        "and ends in a leaf that gives the prediction.",
        approach="The tree splits visits into groups with similar behaviour, like 'no page value and high exit rate'. "
        "Depth is limited to 5 so it stays readable and does not memorise the data.",
        settings="DecisionTreeClassifier(max_depth=5), no scaling",
        scaled=False,
        strengths=[
            "Easy to explain as rules",
            "No scaling needed",
            "Captures non-linear rules and interactions",
        ],
        limits=[
            "Can overfit if it grows deep",
            "A small data change can change the tree",
            "Probabilities are coarse",
        ],
        choose="You need rules that business teams can read and that run without scaling.",
        avoid="You need smooth probabilities or the best possible ranking of visitors.",
    ),
    "KNN": dict(
        principle="Finds the 5 most similar past visits (closest in feature space) and takes a vote.",
        approach="A new visit is compared with training visits. If most of its 5 nearest neighbours bought, it is predicted to buy.",
        settings="StandardScaler, then KNeighborsClassifier(n_neighbors=5)",
        scaled=True,
        strengths=[
            "No training step to speak of",
            "Follows local patterns, whatever their shape",
        ],
        limits=[
            "Needs scaling because it uses distances",
            "Slow to predict on large data",
            "Struggles when many columns add noise",
            "Votes of 5 give rough scores",
        ],
        choose="Similar visits clearly behave alike and the data set is small or medium.",
        avoid="You need fast predictions on large data or many noisy features.",
    ),
    "Naive Bayes": dict(
        principle="Uses Bayes' theorem. It assumes features are independent and each follows a bell curve for each class, "
        "then multiplies the evidence.",
        approach="For each visit it asks how typical these values are for buyers and for non-buyers, and picks the more likely class.",
        settings="StandardScaler, then GaussianNB()",
        scaled=True,
        strengths=[
            "Very fast",
            "Works with little data",
            "Tends to find many positives (high recall)",
        ],
        limits=[
            "Independence assumption is wrong here (many columns are related)",
            "Skewed columns do not follow bell curves",
            "Often over-confident, so precision suffers",
        ],
        choose="Catching as many possible buyers as you can matters more than false alarms.",
        avoid="False alarms are costly or the features are strongly correlated.",
    ),
    "SVM": dict(
        principle="Finds the boundary that separates the two classes with the widest margin. "
        "The RBF kernel lets that boundary be curved.",
        approach="Maps visits into a richer space where buyers and non-buyers can be separated, "
        "and uses only the hardest visits (support vectors) to draw the line.",
        settings="StandardScaler, then SVC(kernel='rbf')",
        scaled=True,
        strengths=["Handles non-linear boundaries", "Works well with many features"],
        limits=[
            "Slower to train on large data",
            "Needs scaling",
            "Hard to explain",
            "No built-in probabilities",
        ],
        choose="You want a flexible boundary and have a medium-sized data set.",
        avoid="You need fast training, explanations, or direct probabilities.",
    ),
}

METRIC_INFO = {
    "Accuracy": "Share of all predictions that were right. With few buyers it can look high even for a weak model.",
    "AUC": "How well the model ranks buyers above non-buyers, over all thresholds. 1 is perfect, 0.5 is random.",
    "Precision": "Of the visitors predicted as buyers, how many really bought. Low precision means many false alarms.",
    "Recall": "Of the real buyers, how many the model found. Low recall means missed sales.",
    "F1": "A balance of precision and recall (their harmonic mean).",
    "MCC": "One fair score for imbalanced data using all four cells of the confusion matrix. 1 is perfect, 0 is random.",
}
