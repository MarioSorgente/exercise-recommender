import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# A simple set of muscle keywords we want to ensure coverage for.
# Feel free to add more or refine these.
MUSCLE_KEYWORDS = [
    "neck", "shoulder", "shoulders", "back", "chest", "arm", "arms",
    "bicep", "tricep", "core", "abs", "abdominals", "hip", "hips",
    "glute", "glutes", "leg", "legs", "hamstring", "calf", "calves"
]

@st.cache_resource
def load_model_and_data():
    """
    1) Load the 'all-MiniLM-L6-v2' SentenceTransformer model.
    2) Read 'Exercise_Database.xlsx' which has columns:
       - Exercise
       - Video
       - Target Muscle Group
       - Primary Equipment
       - Posture
       - Body Region
    3) Create a 'combined_text' for semantic search.
    4) Encode each row into embeddings.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_excel('Exercise_Database.xlsx')

    df['combined_text'] = (
        df['Target Muscle Group'].fillna('') + ' ' +
        df['Body Region'].fillna('') + ' ' +
        df['Exercise'].fillna('') + ' ' +
        df['Posture'].fillna('') + ' ' +
        df['Primary Equipment'].fillna('')
    )

    exercise_embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
    return model, df, exercise_embeddings

def recommend_exercises(user_query, model, df, exercise_embeddings, top_n=10):
    """
    1) Compute similarity between user_query and each row.
    2) Sort by similarity descending.
    3) Skip rows where 'Video' is "NoData" or empty.
    4) Ensure coverage for any muscle keywords the user mentions.
    5) Return up to 'top_n' results with embedded video if available.
    """
    # Create user embedding
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, exercise_embeddings)[0].cpu().numpy()

    # Sort from highest to lowest similarity
    all_indices = similarities.argsort()[::-1]

    # Build a DataFrame with sorted rows
    all_results = df.iloc[all_indices].copy()
    all_results['Similarity'] = similarities[all_indices].round(4)

    # Detect which muscle keywords are in user_query (case-insensitive)
    user_lower = user_query.lower()
    mentioned_muscles = [kw for kw in MUSCLE_KEYWORDS if kw in user_lower]

    # STEP 1: Collect valid-video exercises, skipping "NoData"
    selected_exercises = []  # Will store (index, row) pairs
    for i, row in all_results.iterrows():
        video_str = str(row.get('Video', '')).strip().lower()
        if video_str not in ("nodata", ""):
            selected_exercises.append((i, row))  # store the row index + row data

        if len(selected_exercises) >= 2 * top_n:
            # We won't need more than 2*top_n in the worst case
            break

    # STEP 2: Ensure coverage of mentioned muscles
    # We'll pick exercises in descending similarity, adding them if they help
    # cover new muscles or if we haven't reached top_n yet.
    final_list = []
    used_indices = set()   # to avoid duplicates by index
    covered_muscles = set()

    def covers_muscle(row_data, muscle_keyword):
        # Check if row's combined_text references the muscle
        text = row_data['combined_text'].lower()
        return muscle_keyword in text

    # First pass: add exercises that help cover new muscles, or until we get top_n
    for i, row_data in selected_exercises:
        if len(final_list) >= top_n:
            break

        if i not in used_indices:
            relevant_muscles = [
                m for m in mentioned_muscles if covers_muscle(row_data, m)
            ]
            # We add the row if:
            #  - it covers at least one new muscle not covered yet
            #    OR
            #  - we haven't yet added at least one exercise per mentioned muscle
            #    OR
            #  - we still haven't reached top_n in total
            if any(m not in covered_muscles for m in relevant_muscles) \
               or len(final_list) < len(mentioned_muscles):
                final_list.append((i, row_data))
                used_indices.add(i)
                for m in relevant_muscles:
                    covered_muscles.add(m)

    # Second pass: if we still haven't reached top_n, fill from remainder
    if len(final_list) < top_n:
        for i, row_data in selected_exercises:
            if len(final_list) >= top_n:
                break
            if i not in used_indices:
                final_list.append((i, row_data))
                used_indices.add(i)

    # Convert final_list (index, row_data) to a DataFrame
    # preserving original columns
    rows_for_df = [row_data for (i, row_data) in final_list]
    results = pd.DataFrame(rows_for_df).reset_index(drop=True)

    return results.head(top_n)

def main():
    st.title("AI Exercise Recommender")
    st.write("Describe your pain or target area, then click 'Get Recommendations' to see up to 10 exercises.")

    model, df, exercise_embeddings = load_model_and_data()

    user_input = st.text_input("Type your pain or target area (e.g., 'shoulder and neck'):")

    if st.button("Get Recommendations"):
        if not user_input.strip():
            st.warning("Please enter a description first.")
        else:
            results = recommend_exercises(user_input, model, df, exercise_embeddings, top_n=10)
            
            if results.empty:
                st.error("No exercises found with a valid video. Try a different query.")
            else:
                st.success(f"Top {len(results)} Exercises (up to 10):")

                for idx, row in results.iterrows():
                    # Card-style container
                    st.markdown(
                        f"""
                        <div style="background-color: #f8f9fa; padding: 1rem; margin-bottom: 1rem; 
                                    border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <h4 style="margin-top: 0;">{idx+1}. {row['Exercise']}</h4>
                        """,
                        unsafe_allow_html=True
                    )

                    # Show the video if available
                    video_link = str(row.get('Video', '')).strip()
                    if video_link.lower() not in ("nodata", ""):
                        st.video(video_link)
                    else:
                        st.write("No video available")

                    # Display other fields; omit Similarity
                    st.markdown(
                        f"""
                            <p><strong>Target Muscle Group:</strong> {row.get('Target Muscle Group','')}</p>
                            <p><strong>Primary Equipment:</strong> {row.get('Primary Equipment','')}</p>
                            <p><strong>Posture:</strong> {row.get('Posture','')}</p>
                            <p><strong>Body Region:</strong> {row.get('Body Region','')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()
