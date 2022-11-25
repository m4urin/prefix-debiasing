from src.utils.files import get_folder

for f in get_folder('.cache').get_all_files('.pt'):
    data = f.read()
    data['evaluations'] = {}
    f.write(data)
