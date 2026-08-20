import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
import time
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
import unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import layers, Model

tf.get_logger().setLevel("ERROR")

np.random.seed(123)
tf.random.set_seed(123)

df = pd.read_csv("../CLUSTERING/final_dataset.csv")

features = ["goals_total_per90", "assists_per90",
    "non_penalty_goals_per90", "shots_total_per90", "shots_on_per90",
    "passes_total_per90", "passes_key_per90", "tackles_total_per90",
    "blocks_per90", "interceptions_per90", "duels_total_per90",
    "duels_won_per90", "dribbles_attempts_per90",
    "dribbles_success_per90", "passes_accuracy_pct",
    "shots_on_target_pct", "duels_won_pct", "dribbles_success_pct"]

def normalize(text):
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(text).lower())
        if not unicodedata.combining(c))

player = input("Which base player do you want to use? ")

result = df[df["player_name"].apply(normalize) == normalize(player)]

if result.empty:
    print("Player not found")
    exit()

elif len(result) == 1:
    player_row = result.iloc[0]

else:
    info = ["age_16-20", "age_21-24", "age_25-27", "age_28-31",
        "age_32-35", "age_36-45", "position_D", "position_F",
        "position_M", "nationality_group_AFC", "nationality_group_CAF",
        "nationality_group_CONCACAF", "nationality_group_CONMEBOL",
        "nationality_group_OFC", "nationality_group_UEFA"]

    print("\nWhich one is your player?\n")

    for index, row in result.iterrows():
        active = [column for column in info if row[column] == 1]
        print(f"[{index}] {row['player_name']} - {' - '.join(active)}")

    selected = int(input("\nEnter the index: "))

    if selected not in result.index:
        print("Invalid index.")
        exit()

    player_row = df.loc[selected]

candidates = df.copy()
total_start = time.perf_counter()

if player_row["position_D"] == 1:
    candidates = candidates[(candidates["position_D"] == 1) | (candidates["position_M"] == 1)]

elif player_row["position_F"] == 1:
    candidates = candidates[(candidates["position_F"] == 1) | (candidates["position_M"] == 1)]

candidates = candidates.copy()

candidates["cluster_score"] = ((candidates["kmeans_cluster"] == player_row["kmeans_cluster"]).astype(int)
    + (candidates["agg_cluster"] == player_row["agg_cluster"]).astype(int))

candidates = candidates[(candidates.index != player_row.name) & (candidates["cluster_score"] >= 1)].copy()

if candidates.empty:
    print("No candidates were found.")
    exit()

scaler = StandardScaler()

X_all = scaler.fit_transform(df[features]).astype("float32")

X_candidates = scaler.transform(candidates[features]).astype("float32")

X_target = scaler.transform(pd.DataFrame([player_row[features]], columns=features)).astype("float32")

original_results = candidates[["player_name", "cluster_score"]].copy()

original_results["similarity"] = cosine_similarity(X_target, X_candidates)[0]

original_results = original_results.sort_values("similarity", ascending=False)

def create_pairs(dataframe, X, total=100000):
    pairs, labels, used = [], [], set()
    limit = total // 2
    positives = negatives = attempts = 0
    max_attempts = total * 100

    clusters = dataframe[["kmeans_cluster", "agg_cluster"]].to_numpy()

    while len(labels) < total and attempts < max_attempts:
        attempts += 1

        i, j = np.random.choice(len(dataframe), 2, replace=False)

        pair = tuple(sorted((i, j)))

        if pair in used:
            continue

        matches = np.sum(clusters[i] == clusters[j])

        if matches == 2 and positives < limit:
            label = 1
            positives += 1

        elif matches == 0 and negatives < limit:
            label = 0
            negatives += 1

        else:
            continue

        used.add(pair)
        pairs.append((X[i], X[j]))
        labels.append(label)

    if not pairs:
        raise ValueError("No valid pairs could be created.")

    X1, X2 = zip(*pairs)

    return (np.array(X1, dtype="float32"),
        np.array(X2, dtype="float32"),
        np.array(labels, dtype="float32"))

pairs_start = time.perf_counter()
X1, X2, labels_pair = create_pairs(df, X_all)
pairs_time = time.perf_counter() - pairs_start

encoder_input = layers.Input(shape=(len(features),))
x = layers.Dense(32)(encoder_input)
x = layers.LeakyReLU(negative_slope=0.01)(x)
x = layers.Dense(16)(x)
x = layers.LeakyReLU(negative_slope=0.01)(x)
embedding = layers.Dense(8)(x)
encoder = Model(encoder_input, embedding)

decoder_input = layers.Input(shape=(8,))
x = layers.Dense(16)(decoder_input)
x = layers.LeakyReLU(negative_slope=0.01)(x)
x = layers.Dense(32)(x)
x = layers.LeakyReLU(negative_slope=0.01)(x)
reconstruction = layers.Dense(len(features))(x)

decoder = Model(decoder_input, reconstruction)

encoder_parameters = encoder.count_params()
decoder_parameters = decoder.count_params()
total_parameters = encoder_parameters + decoder_parameters

dataset = tf.data.Dataset.from_tensor_slices((X1, X2, labels_pair)).shuffle(len(labels_pair)).batch(100)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

training_start = time.perf_counter()
for epoch in range(20):
    losses = []

    for batch_1, batch_2, labels in dataset:
        with tf.GradientTape() as tape:
            embedding_1 = encoder(batch_1)
            embedding_2 = encoder(batch_2)

            reconstruction_1 = decoder(embedding_1)
            reconstruction_2 = decoder(embedding_2)

            reconstruction_loss = (
                tf.reduce_mean(tf.square(batch_1 - reconstruction_1))
                + tf.reduce_mean(tf.square(batch_2 - reconstruction_2))) / 2

            distance = tf.sqrt(tf.reduce_sum(
                    tf.square(embedding_1 - embedding_2), axis=1) + 1e-10)

            contrastive_loss = tf.reduce_mean(
                labels * tf.square(distance)
                + (1 - labels) * tf.square(tf.maximum(1 - distance, 0)))

            loss = (0.7 * reconstruction_loss + 0.3 * contrastive_loss)

        variables = (encoder.trainable_variables + decoder.trainable_variables)
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        losses.append(loss.numpy())

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1}/20 - "
            f"Loss: {np.mean(losses):.4f}")

training_time = time.perf_counter() - training_start

target_tensor = tf.convert_to_tensor(X_target)

with tf.GradientTape() as tape:
    tape.watch(target_tensor)
    target_embedding_tensor = encoder(target_tensor)

gradients = tape.jacobian(target_embedding_tensor, target_tensor)

gradients = tf.squeeze(gradients, axis=[0, 2]).numpy()

influence = np.mean(np.abs(gradients), axis=0)

influence = influence / influence.sum() * 100

weights = pd.DataFrame({"variable": features, "local_sensitivity_pct": influence}).sort_values("local_sensitivity_pct", ascending=False)

os.makedirs("weights", exist_ok=True)

filename = normalize(player_row["player_name"]).replace(" ", "_")

weights.to_csv(f"weights/{filename}.txt",
    sep="\t",
    index=False,
    float_format="%.4f")

target_embedding = encoder.predict(X_target,verbose=0)

candidate_embeddings = encoder.predict(X_candidates, verbose=0)

distances = np.linalg.norm(candidate_embeddings - target_embedding,axis=1)

hybrid_results = candidates[["player_name", "cluster_score"]].copy()

hybrid_results["similarity"] = 1 / (1 + distances)

hybrid_results = hybrid_results.sort_values("similarity",ascending=False)

top_original = set(original_results.head(10)["player_name"])

top_hybrid = set(hybrid_results.head(10)["player_name"])

common_players = top_original & top_hybrid

total_time = time.perf_counter() - total_start

print(f"\nTARGET PLAYER: {player_row['player_name']}")

print("\nTOP 10 USING ORIGINAL FEATURES:\n")
print(original_results.head(10).to_string(index=False))

print("\nTOP 10 USING HYBRID EMBEDDINGS:\n")
print(hybrid_results.head(10).to_string(index=False))

print(
    f"\nPlayers appearing in both Top 10: "
    f"{len(common_players)}")

for name in common_players:
    print(name)

print("\nCOMPUTATIONAL RESULTS:\n")
print(f"Dataset players: {len(df)}")
print(f"Candidate players: {len(candidates)}")
print(f"Training pairs: {len(labels_pair)}")
print(f"Encoder parameters: {encoder_parameters}")
print(f"Decoder parameters: {decoder_parameters}")
print(f"Total model parameters: {total_parameters}")
print(f"\nPair creation time: {pairs_time:.4f} seconds")
print(f"Training time: {training_time:.4f} seconds")
print(f"Total execution time: {total_time:.4f} seconds")