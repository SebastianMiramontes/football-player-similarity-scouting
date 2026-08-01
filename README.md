# Football Player Similarity and Scouting System

A machine learning project designed to identify football players with similar statistical profiles across multiple professional leagues.

The system allows the user to select a reference player and generates ranked lists of comparable players using clustering, cosine similarity, and neural-network embeddings. The project is intended as a decision-support tool for football scouting and recruitment.

**Key features:** Outfield player analysis · Per-90 statistics · Player clustering · Cosine similarity · Neural-network embeddings · Interpretable recommendations

---

## Author

Sebastian Miramontes Soto

---

## Project Objective

Football clubs often need to identify replacements, alternatives, or players with characteristics similar to a specific footballer.

This project uses player performance data to reduce the initial scouting pool and generate a shortlist of statistically similar candidates. These recommendations can then be evaluated through video analysis, tactical considerations, financial information, and expert scouting.

The objective is not to automatically determine which player should be signed, but to support the first stage of the recruitment process using data.

---

## Dataset

The dataset contains player performance statistics from 12 professional football leagues across Europe and the Americas.

### Included Leagues

| Country | Competition | Seasons |
|---|---|---|
| Spain | La Liga | 2025 |
| England | Premier League | 2025 |
| Italy | Serie A | 2025 |
| France | Ligue 1 | 2025 |
| Germany | Bundesliga | 2025 |
| Portugal | Primeira Liga | 2025 |
| Netherlands | Eredivisie | 2025 |
| Turkey | Süper Lig | 2025 |
| Mexico | Liga MX | 2025 |
| Argentina | Primera División | 2025–2026 |
| Brazil | Campeonato Brasileiro Série A | 2025–2026 |
| United States | Major League Soccer | 2025–2026 |

Including leagues from different countries and competitive environments allows the model to search for similar players across several football markets.

Match-level data is processed and aggregated to create a statistical profile for each player. The final dataset contains per-90-minute metrics, efficiency percentages, and contextual variables.

---

## Player Profiles

The current version analyzes outfield players who accumulated at least 450 league minutes between January 2025 and June 2026:

- Defenders
- Midfielders
- Forwards

Goalkeepers are excluded because their performance requires a different set of position-specific variables.

Players are represented using both performance statistics and categorical information.

Categorical variables may include:

- Position
- Age group
- Nationality confederation
- League

These variables are converted into numerical format before being used by the models.

---

## Performance Features

The main similarity analysis uses per-90-minute and percentage-based statistics. This makes comparisons more appropriate for players who have played different amounts of minutes.

### Attacking

- Goals per 90
- Assists per 90
- Goals plus assists per 90
- Non-penalty goals per 90
- Shots per 90
- Shots on target per 90
- Shot-on-target percentage

### Passing and Creativity

- Total passes per 90
- Key passes per 90
- Pass accuracy percentage

### Defending

- Tackles per 90
- Blocks per 90
- Interceptions per 90
- Tackles plus interceptions per 90

### Duels and Dribbling

- Total duels per 90
- Duels won per 90
- Duel success percentage
- Dribble attempts per 90
- Successful dribbles per 90
- Dribble success percentage

These variables are intended to represent a player's statistical playing profile rather than only their total production.

---

## Methodology

The project follows four main stages.

### 1. Data Preprocessing

Before applying the models, the data is prepared through:

- Match-level data aggregation
- Missing-value analysis
- Feature selection
- Per-90-minute calculations
- Percentage-based calculations
- Categorical variable encoding
- Numerical feature standardization

### 2. Player Clustering

Two unsupervised clustering methods are used to group players with similar statistical profiles:

- **K-Means Clustering:** divides the players into a predefined number of groups by assigning each player to the nearest cluster center.
- **Agglomerative Clustering:** uses a hierarchical approach that begins with each player in an individual group and gradually combines the most similar groups.

Using both methods makes it possible to compare two different ways of organizing player profiles. K-Means creates groups around central statistical profiles, while Agglomerative Clustering identifies hierarchical relationships between players.

The cluster assigned by each method is used to provide additional context for the similarity rankings. For every recommended player, the system checks whether they share a K-Means cluster, an Agglomerative cluster, or both clusters with the selected player.

### 3. Cosine Similarity

The first recommendation method compares players directly using their standardized original features.

The process is:

1. Select a reference player.
2. Extract the player's standardized statistical profile.
3. Compare the profile with every other player.
4. Calculate cosine similarity.
5. Rank the candidates from highest to lowest similarity.
6. Return the top recommendations.

A cosine similarity score closer to `1` indicates that two players have more similar statistical patterns.

This method acts as the baseline because it is direct and relatively easy to interpret.

### 4. Neural-Network Embeddings

The second method uses an autoencoder neural network.

The encoder transforms each player's original statistics into a smaller numerical representation called an embedding. The decoder then attempts to reconstruct the original player profile.

After training, similarity is calculated between the learned embeddings rather than directly between the original variables.

This approach allows the model to capture nonlinear relationships and combinations of variables that may not be visible through direct statistical comparison.

---

## Model Interpretability

The project includes a local sensitivity analysis using TensorFlow gradients.

For a selected player, the system calculates how changes in each input variable affect the player's neural-network embedding.

The resulting values are converted into percentages and ranked by influence. This helps identify which statistics have the greatest effect on the learned representation of a particular player.

