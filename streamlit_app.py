import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# A simple set of muscle keywords we want to ensure coverage for.
# Feel free to add more or refine these.
MUSCLE_KEYWORDS = [
    "neck", "shoulder", "shoulders", "back", "chest", "arm", "arms", "bicep", "tricep",
    "core", "abs", "abdominals", "hip", "hips", "glute", "glutes", "leg", "legs",
    "hamstring", "calf", "calves"
]

@st.cache_resource
def load_model_and_data():
    """
    1) Loads the SentenceTransformer model: 'all-MiniLM-L6-v2'.
    2) Reads the Excel file 'Exercise_Database.xlsx' with columns:
       - Exercise
       - Video
       - Target Muscle Group
       - Primary Equipment
       - Posture
       - Body Region
    3) Builds a 'combined_text' field from relevant columns.
    4) Encodes each row for semantic search.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_excel('Exercise_Database.xlsx')

    # Combine columns into a single text field
    df['combined_text'] = (
        df['Target Muscle Group'].fillna('') + ' ' +
        df['Body Region'].fillna('') + ' ' +
        df['Exercise'].fillna('') + ' ' +
        df['Posture'].fillna('') + ' ' +
        df['Primary Equipment'].fillna('')
    )

    # Create embeddings for each row
    exercise_embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True)

    return model, df, exercise_embeddings

def recommend_exercises(user_query, model, df, exercise_embeddings, top_n=10):
    """
    1) Computes similarity between the user_query and each row.
    2) Ranks results by similarity.
    3) Skips any row where Video == "NoData" (or empty).
    4) Ensures each muscle in user_query is covered at least once if possible.
    5) Returns up to 'top_n' final results.
    """
    # Create user query embedding
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, exercise_embeddings)[0].cpu().numpy()

    # Sort from highest to lowest similarity
    all_indices = similarities.argsort()[::-1]

    # Build a DataFrame with similarities
    all_results = df.iloc[all_indices].copy()
    all_results['Similarity'] = similarities[all_indices].round(4)

    # Detect which muscle keywords are in user_query (case-insensitive)
    user_lower = user_query.lower()
    mentioned_muscles = [kw for kw in MUSCLE_KEYWORDS if kw in user_lower]

    # We'll collect potential exercises from best to worst, skipping "NoData" for video.
    selected_exercises = []
    used_indices = set()

    # First pass: pick exercises for overall query, skipping "NoData"
    for i, row in all_results.iterrows():
        if len(selected_exercises) >= 2 * top_n:
            # We won't need more than 2*top_n in worst case
            break
        
        video_str = str(row.get('Video', '')).lower()
        if video_str == "nodata" or video_str.strip() == "":
            # Skip if no valid video
            continue

        selected_exercises.append(row)
        used_indices.add(i)

    # We now have a list of valid-video exercises in descending similarity.
    # Next: ensure each mentioned muscle is covered.
    # We'll look for an exercise that references that muscle in 'combined_text'
    # if not already covered.
    def covers_muscle(exercise_row, muscle_keyword):
        # Check if the row's combined_text includes the muscle keyword
        # or the muscle is found in any relevant column.
        text = exercise_row['combined_text'].lower()
        return muscle_keyword in text

    final_list = []
    covered_muscles = set()

    for ex in selected_exercises:
        # Add exercise if it helps cover new muscles or if we don't yet have enough
        if len(final_list) < top_n:
            # Which muscles does this exercise help cover?
            relevant = [m for m in mentioned_muscles if covers_muscle(ex, m)]
            # If it covers at least one muscle that isn't covered yet, great
            # or if we still haven't reached top_n, we add it
            if any(m not in covered_muscles for m in relevant) or len(final_list) < len(mentioned_muscles):
                final_list.append(ex)
                for m in relevant:
                    covered_muscles.add(m)
        else:
            break

    # If we still haven't reached top_n, fill from the remainder
    if len(final_list) < top_n:
        for ex in selected_exercises:
            if ex not in final_list:
                final_list.append(ex)
            if len(final_list) >= top_n:
                break

    # Convert final_list back to a DataFrame
    results = pd.DataFrame(final_list).head(top_n).reset_index(drop=True)

    return results

def main():
    st.title("AI Exercise Recommender")
    st.write("Describe your pain or target area, then click 'Get Recommendations' to see the top exercises.")

    model, df, exercise_embeddings = load_model_and_data()

    user_input = st.text_input("Type your pain or target area (e.g., 'shoulder and neck'):")

    if st.button("Get Recommendations"):
        if not user_input.strip():
            st.warning("Please enter a description first.")
        else:
            # Get up to 10 exercises with coverage + skipping NoData
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
                    if video_link.lower() != "nodata" and video_link != "":
                        st.video(video_link)
                    else:
                        st.write("No video available")

                    # Print other fields. We skip similarity to have a cleaner UI.
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
