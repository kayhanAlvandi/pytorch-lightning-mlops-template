import psycopg

class DBLogger:
    """Logs predictions to a database."""
    def __init__(self, 
                 db_uri: str):
        self.db_uri = db_uri
    def connect(self):
        self.connection = psycopg.connect(self.db_uri)

    
    def log_image_metadata(self, image_metadata: dict):
        for attempt in range(2): # try once, reconnect and retry once
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute("""
                    INSERT INTO image_metadata (plate, well , field, channel , root_path
                    ,file_name, shape_x, shape_y)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (file_name) DO UPDATE SET root_path = EXCLUDED.root_path
                    RETURNING id
                """, (
                    image_metadata["plate"],
                    image_metadata["well"],
                    image_metadata["field"],
                    image_metadata["channel"],
                    image_metadata["root_path"],
                    image_metadata["file_name"],
                    image_metadata["shape_x"],
                    image_metadata["shape_y"]
                    ))
                    image_id = cursor.fetchone()[0]
                self.connection.commit()
                return image_id
            except psycopg.OperationalError:
                print("DB connection lost, reconnecting...")
                self.connection = psycopg.connect(self.db_uri)
        return None