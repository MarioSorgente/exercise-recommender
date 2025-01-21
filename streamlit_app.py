import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model_and_data():
    """
    1) Loads the SentenceTransformer model: 'all-MiniLM-L6-v2'.
    2) Reads the Excel file 'Exercise_Database.xlsx', which must have columns:
       Exercise, Video, Target Muscle Group,
       Primary Equipment, Posture, Body Region.
    3) Builds a 'combined_text' field by concatenating relevant columns.
    4) Encodes each row into an embedding vector for semantic search.
    """
    # 1. Load the transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Read your Excel file
    df = pd.read_excel('Exercise_Database.xlsx')
    
    # 3. Combine columns into a single text field for embedding
    df['combined_text'] = (
        df['Target Muscle Group'].fillna('') + ' ' +
        df['Body Region'].fillna('') + ' ' +
        df['Exercise'].fillna('') + ' ' +
        df['Posture'].fillna('') + ' ' +
        df['Primary Equipment'].fillna('')
    )
    
    # 4. Create embeddings for each exercise row
    exercise_embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
    
    return model, df, exercise_embeddings

def recommend_exercises(user_query, model, df, exercise_embeddings, top_n=5):
    """
    Given a user query (e.g., 'I have neck pain'),
    compute similarity to each exercise and return the top_n matches.
    """
    # Convert the user query into an embedding
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    
    # Calculate cosine similarity between the query and each exercise
    similarities = util.cos_sim(query_embedding, exercise_embeddings)[0].cpu().numpy()
    
    # Sort by highest similarity first
    top_indices = similarities.argsort()[::-1][:top_n]
    
    # Build a DataFrame of the top matches
    results = df.iloc[top_indices][
        ['Exercise', 'Video', 'Target Muscle Group',
         'Primary Equipment', 'Posture', 'Body Region']
    ].copy()
    results['Similarity'] = similarities[top_indices]
    
    return results.reset_index(drop=True)

def main():
    st.title("AI Exercise Recommender")
    st.write("Describe your pain or target area, then click 'Get Recommendations' to see the top exercises.")
    
    # Load everything once
    model, df, exercise_embeddings = load_model_and_data()
    
    # Text input for user query
    user_input = st.text_input("Type your pain or target area (e.g., 'neck and shoulders'):")
    
    # On button click, generate recommendations
    if st.button("Get Recommendations"):
        if not user_input.strip():
            st.warning("Please enter a description first.")
        else:
            results = recommend_exercises(user_input, model, df, exercise_embeddings, top_n=5)
            st.success(f"Top {len(results)} Exercises:")
            
            for idx, row in results.iterrows():
                st.write(f"**{idx+1}.** {row['Exercise']}")
                st.markdown(f"[Video demonstration]({row['Video']})")
                st.write(f"- **Target Muscle Group:** {row['Target Muscle Group']}")
                st.write(f"- **Primary Equipment:** {row['Primary Equipment']}")
                st.write(f"- **Posture:** {row['Posture']}")
                st.write(f"- **Body Region:** {row['Body Region']}")
                st.write(f"- **Similarity:** {row['Similarity']:.4f}")
                st.write("---")

if __name__ == "__main__":
    main()
