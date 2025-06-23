import sqlite3

class Database:
    file_name = "grove_database.db"
    table_name = "grove"
    def __init__(self):
        self.con = sqlite3.connect(self.file_name)
        self.cur = self.con.cursor()
        self.create_table()

    def create_table(self):
        query = f"CREATE TABLE IF NOT EXISTS {self.table_name} ( " \
                f"id INTEGER PRIMARY KEY, " \
                f"a_x REAL NOT NULL, " \
                f"a_y REAL NOT NULL, " \
                f"a_z REAL NOT NULL, " \
                f"g_x REAL NOT NULL, " \
                f"g_y REAL NOT NULL, " \
                f"g_z REAL NOT NULL);"
        self.cur.execute(query)

    def insert(self, **data):
        query = f"INSERT INTO {self.table_name} VALUES (NULL, ?, ?, ?, ?, ?, ?)"
        self.cur.execute(query, list(data.values()))
        self.con.commit()  

    def get(self, id=None):
        extension = f"WHERE id = {id}" if id is not None else ""
        query = f"SELECT * FROM {self.table_name} {extension}"
        res = self.cur.execute(query)
        return res.fetchone() if id is not None else res.fetchall()

    def delete(self, id=None):
        extension = f"WHERE id = {id}" if id is not None else ""
        query = f"DELETE FROM {self.table_name} {extension}"
        res = self.cur.execute(query)
        return res