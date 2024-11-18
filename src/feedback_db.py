# requirements.txt content:
# psycopg2-binary==2.9.9
# pandas==2.1.0
# sqlalchemy==2.0.23
# python-dotenv==1.0.0

import os
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class FeedbackStorage:
    def __init__(self):
        # Get database connection details from environment variables
        self.db_url = os.getenv('DATABASE_URL')
        self.engine = create_engine(self.db_url)

    def initialize_database(self):
        """Create necessary tables if they don't exist"""
        create_tables_sql = """
        -- Create enum types for consistent data
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'usage_frequency') THEN
                CREATE TYPE usage_frequency AS ENUM (
                    'daily', 'weekly', 'once_weekly', 'monthly', 'less_than_monthly'
                );
            END IF;
        END$$;

        -- Main feedback table
        CREATE TABLE IF NOT EXISTS user_feedback (
            feedback_id SERIAL PRIMARY KEY,
            submission_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            name VARCHAR(100),
            email VARCHAR(255),
            usage_frequency usage_frequency,
            mel_rating INTEGER CHECK (mel_rating BETWEEN 1 AND 5),
            goal_setting_rating INTEGER CHECK (goal_setting_rating BETWEEN 1 AND 5),
            progress_tracking_rating INTEGER CHECK (progress_tracking_rating BETWEEN 1 AND 5),
            coaching_feedback_rating INTEGER CHECK (coaching_feedback_rating BETWEEN 1 AND 5),
            likes TEXT,
            suggestions TEXT
        );

        -- Table for improvement areas (many-to-many relationship)
        CREATE TABLE IF NOT EXISTS improvement_areas (
            feedback_id INTEGER REFERENCES user_feedback(feedback_id),
            area VARCHAR(50),
            PRIMARY KEY (feedback_id, area)
        );

        -- Create indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_submission_timestamp 
            ON user_feedback(submission_timestamp);
        CREATE INDEX IF NOT EXISTS idx_email 
            ON user_feedback(email);
        CREATE INDEX IF NOT EXISTS idx_usage_frequency 
            ON user_feedback(usage_frequency);
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(create_tables_sql))
            conn.commit()

    def store_feedback(self, feedback_data):
        """Store feedback in the database"""
        try:
            with self.engine.connect() as conn:
                # Insert main feedback data
                sql = """
                INSERT INTO user_feedback (
                    name, email, usage_frequency, mel_rating,
                    goal_setting_rating, progress_tracking_rating,
                    coaching_feedback_rating, likes, suggestions
                ) VALUES (
                    :name, :email, :usage_frequency, :mel_rating,
                    :goal_setting_rating, :progress_tracking_rating,
                    :coaching_feedback_rating, :likes, :suggestions
                ) RETURNING feedback_id;
                """
                result = conn.execute(text(sql), feedback_data)
                feedback_id = result.scalar()

                # Store improvement areas
                if feedback_data.get('improvement_areas'):
                    for area in feedback_data['improvement_areas']:
                        conn.execute(
                            text("INSERT INTO improvement_areas (feedback_id, area) VALUES (:id, :area)"),
                            {'id': feedback_id, 'area': area}
                        )
                
                conn.commit()
                return feedback_id
        except SQLAlchemyError as e:
            print(f"Error storing feedback: {e}")
            raise

    def get_engagement_stats(self, start_date=None, end_date=None):
        """Get engagement statistics for a given time period"""
        try:
            with self.engine.connect() as conn:
                sql = """
                SELECT 
                    DATE_TRUNC('day', submission_timestamp) as date,
                    COUNT(*) as total_submissions,
                    AVG(mel_rating) as avg_mel_rating,
                    AVG(goal_setting_rating) as avg_goal_rating,
                    AVG(progress_tracking_rating) as avg_progress_rating,
                    AVG(coaching_feedback_rating) as avg_coaching_rating,
                    COUNT(DISTINCT email) as unique_users
                FROM user_feedback
                WHERE (:start_date IS NULL OR submission_timestamp >= :start_date)
                    AND (:end_date IS NULL OR submission_timestamp <= :end_date)
                GROUP BY DATE_TRUNC('day', submission_timestamp)
                ORDER BY date;
                """
                
                df = pd.read_sql(
                    sql,
                    conn,
                    params={'start_date': start_date, 'end_date': end_date}
                )
                return df
            
        except SQLAlchemyError as e:
            print(f"Error getting feedback: {e}")
            raise

    def get_improvement_areas_analysis(self):
        """Analyze most commonly requested improvement areas"""
        sql = """
        SELECT 
            ia.area,
            COUNT(*) as mention_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT feedback_id) FROM improvement_areas), 2) as percentage
        FROM improvement_areas ia
        GROUP BY ia.area
        ORDER BY mention_count DESC;
        """
        
        with self.engine.connect() as conn:
            return pd.read_sql(sql, conn)

    def get_usage_frequency_distribution(self):
        """Analyze usage frequency distribution"""
        sql = """
        SELECT 
            usage_frequency,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM user_feedback
        GROUP BY usage_frequency
        ORDER BY count DESC;
        """
        
        with self.engine.connect() as conn:
            return pd.read_sql(sql, conn)

    def export_feedback_report(self, start_date=None, end_date=None):
        """Generate a comprehensive feedback report"""
        report = {
            'engagement_stats': self.get_engagement_stats(start_date, end_date).to_dict(),
            'improvement_areas': self.get_improvement_areas_analysis().to_dict(),
            'usage_distribution': self.get_usage_frequency_distribution().to_dict(),
            'generated_at': datetime.now().isoformat()
        }
        return report

# Example usage:
def main():
    # Initialize storage
    storage = FeedbackStorage()
    
    # Example feedback data
    feedback_data = {
        'name': 'John Doe',
        'email': 'john@example.com',
        'usage_frequency': 'weekly',
        'mel_rating': 4,
        'goal_setting_rating': 5,
        'progress_tracking_rating': 4,
        'coaching_feedback_rating': 5,
        'improvement_areas': ['ui', 'notifications'],
        'likes': 'Great coaching experience',
        'suggestions': 'More customization options'
    }
    
    # Store feedback
    feedback_id = storage.store_feedback(feedback_data)
    
    # Generate reports
    stats = storage.get_engagement_stats()
    improvements = storage.get_improvement_areas_analysis()
    usage = storage.get_usage_frequency_distribution()
    
    # Export comprehensive report
    report = storage.export_feedback_report()
    
    return report

if __name__ == "__main__":
    main()