from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("features_30_sec.csv")
TARGET_COLUMN = "label"
IGNORED_COLUMNS = ["filename", TARGET_COLUMN]


st.set_page_config(
    page_title="Music Genre Classifier",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Could not find {DATA_PATH}. Keep it in the same folder as app.py.")
        st.stop()

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(df: pd.DataFrame):
    feature_columns = [col for col in df.columns if col not in IGNORED_COLUMNS]
    x = df[feature_columns]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, feature_columns, accuracy


def prediction_table(model, row: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(row)[0]
    classes = model.classes_
    result = pd.DataFrame(
        {
            "Genre": classes,
            "Confidence": probabilities,
        }
    )
    return result.sort_values("Confidence", ascending=False).reset_index(drop=True)


def editable_features(df: pd.DataFrame, feature_columns: list[str], selected_index: int) -> pd.DataFrame:
    base_values = df.loc[selected_index, feature_columns].copy()

    with st.expander("Edit audio features"):
        st.caption("Adjust values if you want to test a custom feature profile.")
        edited_values = {}
        columns = st.columns(3)

        for index, feature in enumerate(feature_columns):
            series = df[feature]
            value = float(base_values[feature])
            edited_values[feature] = columns[index % 3].number_input(
                feature,
                value=value,
                min_value=float(series.min()),
                max_value=float(series.max()),
                step=float((series.max() - series.min()) / 100) or 0.01,
                format="%.6f",
            )

    return pd.DataFrame([edited_values], columns=feature_columns)


def main() -> None:
    df = load_data()
    model, feature_columns, accuracy = train_model(df)

    st.title("Music Genre Classifier")
    st.write("A Streamlit version of the app using your GTZAN feature dataset.")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Songs", f"{len(df):,}")
    metric_cols[1].metric("Genres", df[TARGET_COLUMN].nunique())
    metric_cols[2].metric("Model accuracy", f"{accuracy:.1%}")

    predict_tab, explore_tab = st.tabs(["Predict", "Explore Data"])

    with predict_tab:
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Choose a song")
            genre_filter = st.selectbox(
                "Filter by genre",
                ["All"] + sorted(df[TARGET_COLUMN].unique().tolist()),
            )

            filtered_df = df if genre_filter == "All" else df[df[TARGET_COLUMN] == genre_filter]
            options = filtered_df.index.tolist()
            labels = {
                index: f"{df.loc[index, 'filename']} ({df.loc[index, TARGET_COLUMN]})"
                for index in options
            }
            selected_index = st.selectbox(
                "Song from dataset",
                options,
                format_func=lambda index: labels[index],
            )

            input_row = editable_features(df, feature_columns, selected_index)

            st.info(
                "Audio upload prediction needs feature extraction libraries such as librosa. "
                "They are not installed in this environment, so this app predicts from the "
                "features already present in your CSV."
            )

        with right:
            st.subheader("Prediction")
            predicted_genre = model.predict(input_row)[0]
            probabilities = prediction_table(model, input_row)

            st.success(f"Predicted genre: **{predicted_genre}**")
            st.dataframe(
                probabilities.assign(Confidence=lambda data: data["Confidence"].map("{:.2%}".format)),
                hide_index=True,
                use_container_width=True,
            )
            st.bar_chart(probabilities.set_index("Genre"))

    with explore_tab:
        st.subheader("Dataset")

        selected_genres = st.multiselect(
            "Genres",
            sorted(df[TARGET_COLUMN].unique().tolist()),
            default=sorted(df[TARGET_COLUMN].unique().tolist()),
        )
        visible_df = df[df[TARGET_COLUMN].isin(selected_genres)] if selected_genres else df

        st.dataframe(visible_df, use_container_width=True, hide_index=True)
        st.subheader("Genre Counts")
        st.bar_chart(visible_df[TARGET_COLUMN].value_counts().sort_index())


if __name__ == "__main__":
    main()
