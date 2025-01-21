import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

################################
# AI MODELING EXPLANATION BELOW
################################
@st.cache_resource
def load_model_and_data():
    """
    1) We load a SentenceTransformer model called 'all-MiniLM-L6-v2'.
       This is a small, efficient transformer model that converts text into embeddings (vectors).
    2) We read the Excel file: Exercise_Database.xlsx, which must be in the same repo.
    3) We build a 'combined_text' field by concatenating columns like
       Target Muscle Group, Body Region, Exercise, Posture, Primary Equipment.
       This ensures the model sees all relevant context.
    4) We encode every row's 'combined_text' into a vector using the model.
    """
    # 1. Load a small, efficient model for semantic search
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Load Excel database (it must be in the same directory)
    df = pd.read_excel('Exercise_Database.xlsx')

    # 3. Combine relevant columns into one text for embedding
    df['combined_text'] = (
        df['Target Muscle Group'].fillna('') + ' ' +
        df['Body Region'].fillna('') + ' ' +
        df['Exercise'].fillna('') + ' ' +
        df['Posture'].fillna('') + ' ' +
        df['Primary Equipment'].fillna('')
    )

    # 4. Encode each row's text into a vector
    exercise_embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
    
    # Return them for later use
    return model, df, exercise_embeddings

def recommend_exercises(user_query, model, df, exercise_embeddings, top_n=5):
    """
    Takes a user query (like 'I have neck pain'),
    computes semantic similarity with each exercise row,
    and returns the top_n matches.
    """
    # Encode the user’s query into a vector
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    
    # Compare query vector to each exercise embedding using cosine similarity
    similarities = util.cos_sim(query_embedding, exercise_embeddings)[0].cpu().numpy()
    
    # Sort exercises from most similar to least similar
    top_indices = similarities.argsort()[::-1][:top_n]
    
    # Build a small dataframe of top matches
    results = df.iloc[top_indices][
        ['Exercise', 'Short YouTube Demonstration', 'Target Muscle Group', 'Body Region']
    ].copy()
    results['Similarity'] = similarities[top_indices]
    
    return results.reset_index(drop=True)

################################
# STREAMLIT APPLICATION CODE
################################
def main():
    st.title("AI Exercise Recommender")
    st.write("Type a description of your pain or target area and click 'Get Recommendations'.")

    # Load model & data only once (cached)
    model, df, exercise_embeddings = load_model_and_data()

    user_input = st.text_input("Describe your pain or target area (e.g., 'neck and shoulders'): ")

    if st.button("Get Recommendations"):
        if not user_input.strip():
            st.warning("Please enter a description first.")
        else:
            # Get top 5 exercise recommendations
            results = recommend_exercises(user_input, model, df, exercise_embeddings, top_n=5)
            
            st.success(f"Top {len(results)} Recommendations:")
            for idx, row in results.iterrows():
                st.write(f"**{idx+1}.** {row['Exercise']}")
                st.write(f"• **YouTube Link:** {row['Short YouTube Demonstration']}")
                st.write(f"• **Target Muscle Group:** {row['Target Muscle Group']}")
                st.write(f"• **Body Region:** {row['Body Region']}")
                st.write(f"• **Similarity Score:** {row['Similarity']:.4f}")
                st.write("---")

if __name__ == "__main__":
    main()
