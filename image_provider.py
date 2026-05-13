import os

class ImageProvider:
    def __init__(self, folder_path):
        all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        # Group by timestamp (the part before the underscore)
        self.pairs = {}
        for f in all_files:
            timestamp = f.split('_')[0]
            timestamp = timestamp[:-3] # Remove seconds and minutes to prevent misalignments
            if timestamp not in self.pairs:
                self.pairs[timestamp] = {'dim': None, 'bright': None}
            
            if 'dim' in f: self.pairs[timestamp]['dim'] = os.path.join(folder_path, f)
            if 'bright' in f: self.pairs[timestamp]['bright'] = os.path.join(folder_path, f)
        
        self.timestamps = sorted(list(self.pairs.keys()))

    def get_pair(self, index):
        if index >= len(self.timestamps): return None
        return self.pairs[self.timestamps[index]]

    def __len__(self):
        return len(self.timestamps)