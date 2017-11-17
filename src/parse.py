import json
import os
import sys

class Parser:
    def __init__(self, path):
        self._path = path
        self._data = []
        self._list_path = self.get_list_paths()
        self._person_info, self._walk_info = self.init_info_dicts()
        self._right_features = self.init_right_deatures()

    '''
    инициализация словарей для конвертирования выборки
    '''
    def init_info_dicts(self):
        person_info = dict(sedentary = 1, medium = 2, active = 3,male = 1, female = 0,yes = 1, no = 0,left = 0, right = 1)
        walk_info = dict(hungry = 0, full = 1, overwrought = 2, barefoot = 0, sport = 1, without_heel = 2, low_heel = 3, middle_heel = 4, height_heel = 5, spike_heel = 6,\
        walk = 0, fast_walk = 1, run = 2, stairs_up = 3, stairs_down = 4, free = 0,left_hand = 1, right_hand = 2, bag = 3, left_shoulder = 4, right_shoulder = 5, left_shoulder_across = 6, right_shoulder_across = 7,\
        sober = 0, drunk = 1)
        return person_info, walk_info

    def init_right_deatures(self):
        right_futures = dict(sedentary = 'Малоподвижный', medium = 'Среднии', active = 'Активный',\
        hungry = 'Голодный', full = 'Сытый', overwrought = 'Переевший',\
        barefoot = 'Без обуви', sport = 'Спортивная', without_heel = 'Без каблука', low_heel = 'Невысокий каблук', middle_heel = 'Средний каблук', height_heel = 'Высокий каблук', spike_heel = 'Каблук-шпилька',\
        walk = 'Ходьба', fast_walk = 'Быстрая ходьба', run = 'Бег', stairs_up = 'Подъем по лестнице вверх', stairs_down = 'Подъем по лестнице вниз',\
        free = 'Без дополнительного веса', left_hand = 'Вес в левой руке', right_hand = 'Вес в правой руке', bag = 'Вес за спиной (рюкзак)', left_shoulder = 'Вес на левом плече', right_shoulder = 'Вес на правом плече', left_shoulder_across = 'Лямка на левом, вес справа', right_shoulder_across = 'Лямка на правом, вес слева',\
        sober = 'Трезвый', drunk = 'Пьяный', yes = 'yes', no = 'no', left = 'left', right = 'right', male = 'male', female = 'female')
        return right_futures

    '''
    замена русскоязычных и составных фраз в выборке
    '''
    def edit_features(self):
        rang_params = ['name', 'age', 'feet size', 'height', 'weight', 'pathology', 'trauma']
        to_int_params = ['age', 'feet size', 'height', 'weight']
        for d in self._data:
            for key, value in list(d['person_info'].items()):
                if key in to_int_params:
                    d['person_info'][key] = int(value)
                if key not in rang_params:
                    right_key = list(self._right_features.keys())[list(self._right_features.values()).index(value)]
                    right_value = self._person_info[right_key]
                    d['person_info'][key] = right_value

            for key, value in list(d['walk_info'].items()):
                right_key = list(self._right_features.keys())[list(self._right_features.values()).index(value)]
                right_value = self._walk_info[right_key]
                d['walk_info'][key] = right_value



    '''
    возвращает имена всех директорий с наборами данных
    '''
    def get_list_paths(self):
        try:
            os.chdir('../data/')
        except:
            print('Неправильная структура файлов. Отсутсвует директория ../data/')
            exit()
        _list = os.listdir()
        if(len(_list) == 0):
            print('Папка с выборкой пуста')
            exit()
        return _list

    '''
    вывод рейтинга на экран в строковом виде
    '''
    def print_rating(self, lables, values):
        for i in range(len(lables)):
            print(str(lables[i]) + ":" + str(values[i]))

    '''
    отображает рейтинг по выборке по заданному критерию в виде гистаграммы
    '''
    def view_rating(self, lable1, lable2, data):
        x = []
        for fd in data:
            x.append(fd[lable1][lable2])
        from collections import Counter
        import numpy as np
        import matplotlib.pyplot as plt
        labels, values = zip(*Counter(x).items())
        indexes = np.arange(len(labels))
        #self.print_rating(values, labels)
        width = 1
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels, rotation=90, fontsize = 10, va='bottom', ha='right')
        plt.show()


    '''
    парсит всю выборку и возвращает список в всех наборов - 1 элемент = 1 снятие движения
    return list[dict[data : list -> dict, person_info : dict, walk_info : dict]]
    '''
    def parse_path(self, min_size_dataset):
        self.delete_all_info_files()
        self.edit_features()
        for path in self._list_path:
            os.chdir(self._path)
            os.chdir(path)
            file_names = os.listdir()
            for i in range(len(file_names)):
                with open(file_names[i]) as f:
                    person_info = json.loads(f.readline())
                    walk_info = json.loads(f.readline())
                    tmp = f.readlines()
                    if (len(tmp) <= min_size_dataset):
                        print('Слишком короткая выборка:' + " "+ person_info['name'] + ", "+ file_names[i]  + ", size = "+ str(len(tmp)))
                    full_file = []
                    for fd in tmp:
                        txyz = fd.replace('\n', '').split('\t')
                        four = dict(time = int(txyz[0]), x = float(txyz[1]), y = float(txyz[2]), z = float(txyz[3]))
                        full_file.append(four)
                    self._data.append(dict(person_info = person_info, walk_info = walk_info, data = full_file))
        return self._data


    '''
    преобрпзование признаков в ранговый вид
    '''
    def feature_convert(self):
        return True
    '''
    удаляет все info файлы во всех директориях я выборкой
    '''
    def delete_all_info_files(self):
        for path in self._list_path:
            os.chdir(self._path)
            os.chdir(path)
            try:
                os.remove('info.txt')
            except:
                continue

    '''
    удаление n последних элементов из всей выборки (Необходимо ещё определиться, сколько именно надо вырезать с конца)
    '''
    def delete_from_back(self, n):
        count = 0
        for d in self._data:
            for i in range(len(d['data']) - n, len(d['data'])):
                try:
                    d['data'].pop()
                except:
                    pass
                    #print("Слишком маленькая выборка, не удается обрезать хвост: " + d['person_info']['name'] + "-" + str(len(d['data'])))
                    #exit()

    '''
    разделение выборки на временные отрезки на вход подаётся один элемент выборки и времянной промежуток
    обратно возвращается столько кусков, сколько влезет
    '''
    def split_only_data_element(self, data_element, num_parts):
        title = [data_element['person_info'], data_element['walk_info']]
        parts = []
        tmp = []
        for i in range(len(data_element['data'])):
            tmp.append(data_element['data'][i])
            if len(tmp) == num_parts:
                value = dict(person_info = title[0], walk_info = title[1], data = tmp)
                parts.append(value)
                del tmp
                tmp = []
        return parts

    '''
    разделение всей выборки на времянные участки
    необходимо указать какого размера будет участок (количественно)
    вернёт разделенныю выборку, но в поле self._data всё равно останется выборка не разделённая
    '''
    def get_split_database(self, parts_size):
        split_data = []
        for d in self._data:
            split_data += self.split_only_data_element(d, parts_size)
        return split_data


if __name__ == "__main__":
    p = Parser('/home/vadim/hackatones/medhack/data/')
    data = p.parse_path(100)
    #p.view_rating('person_info', 'pathology', data)
    p.delete_from_back(20)
    split_data = p.get_split_database(10)
    p.edit_features()
    print(split_data[0]['person_info'])
    print(split_data[0]['walk_info'])
    #p.view_rating('walk_info', 'influence', split_data)
