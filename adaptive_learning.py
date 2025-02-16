# adaptive_learning.py
import pandas as pd
import os

def parse_feedback(log_file):
    """Parse feedback logs into a DataFrame."""
    if not log_file or not os.path.exists(log_file):
        print("No feedback log available.")
        return None
    feedback_df = pd.read_csv(log_file, names=["timestamp", "track_name", "feedback"])
    return feedback_df

def update_user_preferences(feedback_df):
    """Update user preferences based on feedback."""
    if feedback_df is None or feedback_df.empty:
        print("No feedback data available for adaptation.")
        return {}

    user_preferences = {}
    for _, row in feedback_df.iterrows():
        track_name = row["track_name"]
        feedback = row["feedback"]

        if "increase volume" in feedback:
            user_preferences[track_name] = user_preferences.get(track_name, {})
            user_preferences[track_name]["volume"] = user_preferences[track_name].get("volume", 1.0) + 0.1
        elif "decrease volume" in feedback:
            user_preferences[track_name] = user_preferences.get(track_name, {})
            user_preferences[track_name]["volume"] = max(0.0, user_preferences[track_name].get("volume", 1.0) - 0.1)

    print("User preferences updated:", user_preferences)
    return user_preferences

if __name__ == "__main__":
    # Example usage
    feedback_data = parse_feedback("feedback_log.csv")
    user_preferences = update_user_preferences(feedback_data)