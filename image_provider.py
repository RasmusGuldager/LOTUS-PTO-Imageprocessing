import os

class ImageProvider:
    def __init__(self, folder_path, chunk_size=5):
        self.files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
        self.chunk_size = chunk_size
        self.current_idx = 0

    def get_next_chunk(self) -> list[str] | None:
        if self.current_idx >= len(self.files) - self.chunk_size:
            return None 
        
        chunk = self.files[self.current_idx:self.current_idx + self.chunk_size]
        self.current_idx += self.chunk_size
        return chunk