import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from knowledge_loader import get_cached_embeddings

WATCH_FILE = "department_knowledge.txt"
WATCH_FILE_PATH = os.path.abspath(WATCH_FILE)

class KnowledgeFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_modified = os.path.getmtime(WATCH_FILE_PATH)

    def on_modified(self, event):
        
        if os.path.abspath(event.src_path) == WATCH_FILE_PATH:
            current_time = os.path.getmtime(WATCH_FILE_PATH)
            if current_time != self.last_modified:
                print("📄 department_knowledge.txt modified. Reloading cache...")
                get_cached_embeddings()
                self.last_modified = current_time
                

def start_file_watcher():
    event_handler = KnowledgeFileHandler()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    print("👁️ Monitoring department_knowledge.txt for changes...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_file_watcher()
