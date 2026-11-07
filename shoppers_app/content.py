"""Static teaching text used by the app (feature meanings, model explanations, metric meanings)."""

FEATURE_INFO = [
    (
        "Administrative",
        "Count",
        "How many account pages the visitor opened, like login or account settings.",
        "Shows if the visitor was busy with their account. Almost half of all visits open none.",
    ),
    (
        "Administrative_Duration",
        "Seconds",
        "The total time spent on those account pages.",
        "A long time here can mean the visitor was checking an order or filling in their details.",
    ),
    (
        "Informational",
        "Count",
        "How many info pages the visitor opened, like About us, Contact or Help.",
        "Someone reading help pages may still be making up their mind. About 78% of visits open none.",
    ),
    (
        "Informational_Duration",
        "Seconds",
        "The total time spent on those info pages.",
        "The same idea as above, but counted in time instead of pages.",
    ),
    (
        "ProductRelated",
        "Count",
        "How many product pages the visitor looked at.",
        "More product pages usually means the visitor is really shopping around. A typical visit sees about 18.",
    ),
    (
        "ProductRelated_Duration",
        "Seconds",
        "The total time spent on product pages.",
        "A strong sign of interest. But a few visits last for hours, so the numbers are very uneven.",
    ),
    (
        "BounceRates",
        "Ratio (0 to 1)",
        "How often people land on a page and leave straight away, without clicking anything else. "
        "It is averaged over the pages in the visit.",
        "A high value means people leave fast, so a purchase is unlikely.",
    ),
    (
        "ExitRates",
        "Ratio (0 to 1)",
        "How often a page is the very last one people see before they leave. "
        "It is averaged over the pages in the visit.",
        "A high value means the visitor is likely to leave soon, so a purchase is less likely.",
    ),
    (
        "PageValues",
        "Value (money)",
        "A score for how much money the visited pages usually bring in, based on past sales.",
        "Usually the strongest clue. But be careful: it is worked out from sales, so it partly knows the answer. "
        "Only about 4 in 100 visits with no page value ended in a purchase. With a page value, more than half did.",
    ),
    (
        "SpecialDay",
        "Number from 0 to 1",
        "How close the visit was to a special day, like Valentine's Day or Mother's Day. "
        "0 means no special day is near. 1 is the peak day.",
        "You might expect more sales near these days, but here it is the opposite. "
        "Visits close to a special day bought less (about 6%, against 17% on other days).",
    ),
    (
        "Month",
        "Category",
        "The month of the visit.",
        "Shows the time of year. May and November have the most visits. "
        "November has the best purchase rate (about 25%) and February the worst (under 2%).",
    ),
    (
        "OperatingSystems",
        "Category code",
        "The visitor's operating system. The data only gives a number, not the name.",
        "May hint at the kind of device or customer. The numbers are labels, not amounts.",
    ),
    (
        "Browser",
        "Category code",
        "The visitor's browser. The data only gives a number, not the name.",
        "May hint at the kind of device the visitor uses. The numbers are labels, not amounts.",
    ),
    (
        "Region",
        "Category code",
        "The area where the visitor is located. The data only gives a number, not the place.",
        "People in different regions can shop in different ways.",
    ),
    (
        "TrafficType",
        "Category code",
        "How the visitor reached the site. The data only gives a number for each source, not its name.",
        "Different sources can bring different kinds of visitors, so some codes buy more than others.",
    ),
    (
        "VisitorType",
        "Category",
        "New_Visitor, Returning_Visitor or Other.",
        "A bit surprising: new visitors bought more often than returning ones (about 25%, against 14%).",
    ),
    (
        "Weekend",
        "True / False",
        "True if the visit was on a weekend.",
        "Weekend visits bought slightly more often (about 17%, against 15%).",
    ),
    (
        "Revenue",
        "True / False (the target)",
        "True if the visit ended in a purchase.",
        "This is what the models try to guess. Only about 16 in 100 visits are True.",
    ),
]

# New columns we build from the old ones
FEATURE_ENG = {
    "TotalPages": dict(
        what="All the pages seen in one visit: account, info and product pages added together.",
        formula="Administrative + Informational + ProductRelated",
        why="These three counts all tell the same story: how much the person browsed. "
        "One total is simpler for a model to use.",
        cols=["Administrative", "Informational", "ProductRelated"],
    ),
    "TotalDuration": dict(
        what="The total time on the site, in seconds.",
        formula="Administrative_Duration + Informational_Duration + ProductRelated_Duration",
        why="One number for 'how long did they stay' instead of three separate times.",
        cols=[
            "Administrative_Duration",
            "Informational_Duration",
            "ProductRelated_Duration",
        ],
    ),
    "AvgTimePerPage": dict(
        what="The average number of seconds spent on each page.",
        formula="TotalDuration / TotalPages  (0 when TotalPages is 0)",
        why="Tells slow readers (few pages, long time) apart from fast skimmers (many pages, short time).",
        cols=["TotalDuration", "TotalPages"],
    ),
    "HasPageValue": dict(
        what="1 if the visit saw pages with some value, 0 if not.",
        formula="1 if PageValues > 0 else 0",
        why="Most visits (about 78%) have no page value, so a simple yes or no is easy to use. "
        "And it matters: with a page value, more than half of visits ended in a purchase. Without one, only about 4%.",
        cols=["PageValues"],
    ),
    "IsReturning": dict(
        what="1 if the visitor has been here before, 0 if not.",
        formula="1 if VisitorType == 'Returning_Visitor' else 0",
        why="Turns the visitor type into a simple yes or no. In this data, returning visitors actually bought less often.",
        cols=["VisitorType"],
    ),
    "IsHolidaySeason": dict(
        what="1 if the visit was in November or December, 0 for any other month.",
        formula="1 if Month in ['Nov', 'Dec'] else 0",
        why="November has the best purchase rate in this data (about 25%) and December is busy too. "
        "This flag lets the model use that time-of-year pattern.",
        cols=["Month"],
    ),
    "log_ProductRelated_Duration": dict(
        what="A squeezed version of the time spent on product pages (the log of 1 + the time).",
        formula="log(1 + ProductRelated_Duration)",
        why="A few visits last for hours, which stretches the column a lot. "
        "The log shrinks those huge values so they don't drown out everyone else.",
        cols=["ProductRelated_Duration"],
    ),
}

# The five models, explained simply
MODEL_INFO = {
    "Logistic Regression": dict(
        principle="Gives every clue a weight (a push up or down) and adds them together into a chance of buying. "
        "The line it draws between buyers and non-buyers is straight.",
        approach="Clues like PageValues push the chance up. Clues like ExitRates push it down. "
        "If the final chance is above 50%, the visit is called a buyer.",
        settings="StandardScaler, then LogisticRegression(max_iter=1000)",
        scaled=True,
        strengths=[
            "Fast and steady",
            "Easy to explain, because you can read each clue's weight",
            "Gives sensible chances, not just yes or no",
        ],
        limits=[
            "Can only draw straight boundaries",
            "Misses clues that work together, unless you add them by hand",
        ],
        choose="You want a simple, easy-to-explain starting point.",
        avoid="The pattern is curvy or full of clues that depend on each other.",
    ),
    "Decision Tree": dict(
        principle="Plays a game of yes or no questions, like 'Is PageValues above 0?'. "
        "Each answer leads down a branch until it reaches a final answer.",
        approach="It sorts visits into small groups that act alike, for example 'no page value and a high exit rate'. "
        "We stop it at 5 levels so it stays easy to read and doesn't just memorise the data.",
        settings="DecisionTreeClassifier(max_depth=5), no scaling",
        scaled=False,
        strengths=[
            "Easy to explain as simple rules",
            "No scaling needed",
            "Can pick up curvy patterns and clues that work together",
        ],
        limits=[
            "Can memorise the data if it grows too deep",
            "A small change in the data can give a very different tree",
            "Its chances are rough, not smooth",
        ],
        choose="You want rules that anyone on the team can read.",
        avoid="You need smooth chances, or the best possible ranking of visitors.",
    ),
    "KNN": dict(
        principle="Looks for the 5 past visits that are most like the new one, then goes with the majority.",
        approach="Imagine asking 5 similar shoppers what they did. "
        "If most of them bought, it predicts that this visitor will buy too.",
        settings="StandardScaler, then KNeighborsClassifier(n_neighbors=5)",
        scaled=True,
        strengths=[
            "Simple idea, with almost no training: it just remembers the data",
            "Learns local patterns of any shape",
        ],
        limits=[
            "Needs scaling, because it measures distances",
            "Gets slow when there is a lot of data",
            "Many unhelpful columns can confuse it",
            "Only 5 votes, so its scores are rough (0%, 20%, 40%...)",
        ],
        choose="Similar visits really do act alike, and your data is small or medium.",
        avoid="You need fast answers on big data, or you have many noisy columns.",
    ),
    "Naive Bayes": dict(
        principle="Uses probability. It treats every clue as separate from the others and asks: "
        "how common are these values among buyers, and how common among non-buyers? "
        "It assumes each clue's values form a bell curve.",
        approach="For each visit it compares 'looks like a buyer' with 'looks like a non-buyer' "
        "and picks the one that is more likely.",
        settings="StandardScaler, then GaussianNB()",
        scaled=True,
        strengths=[
            "Very fast",
            "Works even with little data",
            "Finds many of the buyers (high recall)",
        ],
        limits=[
            "It treats clues as unrelated, but here many are linked (like pages and time spent)",
            "Skewed columns don't really form bell curves",
            "It is often too sure of itself, so it raises many false alarms (low precision)",
        ],
        choose="Missing a buyer costs you more than a false alarm.",
        avoid="False alarms are costly, or many of your clues are closely linked.",
    ),
    "SVM": dict(
        principle="Draws the boundary that leaves the widest gap between buyers and non-buyers. "
        "The RBF setting lets that boundary bend and curve.",
        approach="It moves the visits into a richer space where the two groups are easier to pull apart. "
        "Only the trickiest cases near the boundary (called support vectors) shape the line.",
        settings="StandardScaler, then SVC(kernel='rbf')",
        scaled=True,
        strengths=[
            "Can draw curved boundaries",
            "Copes well with many columns",
        ],
        limits=[
            "Slow to train on big data",
            "Needs scaling",
            "Hard to explain why it decided what it did",
            "Doesn't give chances out of the box",
        ],
        choose="You want a flexible model and your data is medium-sized.",
        avoid="You need quick training, clear explanations or ready-made chances.",
    ),
}

# The six scores, explained simply
METRIC_INFO = {
    "Accuracy": "Out of all predictions, how many were right. Careful: about 84 in 100 visits don't buy, so a model that always says 'no purchase' would already score about 84%.",
    "AUC": "How well the model puts real buyers above non-buyers when it ranks all the visits. 1 is perfect and 0.5 is no better than a coin flip.",
    "Precision": "When the model says 'this visitor will buy', how often is it right? Low precision means lots of false alarms.",
    "Recall": "Out of all the visitors who really bought, how many did the model catch? Low recall means missed sales.",
    "F1": "One score that balances precision and recall. It is only high when both are decent.",
    "MCC": "One fair score for data like this, where buyers are rare. It looks at every kind of right and wrong answer. 1 is perfect, 0 is no better than guessing, and below 0 is worse than guessing.",
}
