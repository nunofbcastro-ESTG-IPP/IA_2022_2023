from kaggle.api.kaggle_api_extended import KaggleApi
from matplotlib.figure import Figure
from Solution import Solution
import zipfile
import os
    
def DownloadDataset(competition_name = 'lei-ia-2223-battleground', path='datasets'):
    api = KaggleApi()
    api.authenticate()
    
    
    destination_folder = Path("", path)
    
    files = api.competition_list_files(competition_name)
    
    for file in files:
        file_string = str(file)
        file_path = Path(file_string, "datasets")
        file_path_zip = f'{file_path}.zip'
        
        if os.path.exists(file_path):
            continue
        
        api.competition_download_file(competition_name, file_string, path=destination_folder)
        
        if not os.path.exists(file_path_zip):
            continue
        
        with zipfile.ZipFile(file_path_zip, 'r') as zipref:
            zipref.extractall(f'{destination_folder}')
            
        os.remove(file_path_zip)

def Path(file: str, folder: str)->str:
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), folder, file)

def SaveLog(pop: list[Solution], hist: list, file_name: str):    
    with open(Path(f'sol_{file_name}.log', 'logs'), 'w') as f:
        pop.sort(key=lambda sol: sol.fitness, reverse=True)
        for sol in pop:
            f.write(f'{sol.__str__()}\n')
    
    SaveLogHist(hist, Path(f'hist_{file_name}.log', 'logs'))

def SaveLogHist(hist: list, path: str):
    with open(path, 'w') as f:
        for i in reversed(range(len(hist))):
            f.write(f'Generation {i}:\n')
            for solutionType in hist[i]:
                hist[i][solutionType].sort(key=lambda sol: sol.fitness, reverse=True)
                for sol in hist[i][solutionType]:
                    f.write(f'{sol.__str__()}\n')
                f.write('\n')
            f.write('\n-------------------------------------------------------\n')
            
def savePlot(fig: Figure, file_name: str):
    fig.savefig(Path(f'polt_{file_name}.png', 'logs'))