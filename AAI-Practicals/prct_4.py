import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "User": ["A", "A", "A", "B", "B", "C", "C", "D"],
    "Item": ["I1", "I2", "I3", "I1", "I4", "I2", "I3", "I4"],
    "Rating": [5, 4, 3, 5, 2, 4, 5, 4]
}

df = pd.DataFrame(data)

user_item_matrix = df.pivot_table(
    index="User",
    columns="Item",
    values="Rating"
).fillna(0)

similarity = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(
    similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

def recommend(user, n=2):
    scores = similarity_df[user].drop(user)
    other_users = user_item_matrix.drop(user)

    weighted_ratings = np.dot(scores.values, other_users.values)
    recommendations = pd.Series(weighted_ratings, index=other_users.columns)

    return recommendations.sort_values(ascending=False).head(n)

print("Recommendations for User A:")
print(recommend("A"))
