# stanford_drone_segmentation

Необходимо сегментировать изображение по данным, размеченным bounding-box'ами.
1. Подход №1: Находим похожий датасет с разметкой для сегментации, обучаем на нем сеть для сегментации, прогоняем на наших изображениях. Такой датасет нашелся - https://www.kaggle.com/bulentsiyah/semantic-drone-dataset. В качестве baseline можно взять какую-то U-net. Скорее всего работать будет не очень хорошо, из-за того, что на нашем датасете масштаб намного меньше и качество изображений гораздо хуже
2. Подход 2 - обучаем модель непосредственно на данных текущего датасета. Применяем Weakly Supervised Semantic Segmentation, например модель Puzzle-CAM (https://arxiv.org/pdf/2101.11253v3.pdf), пример на github - https://github.com/OFRIN/PuzzleCAM

3. Подход 3 - поскольку объекты на изображениях достаточно мелкие, заменим боксы, получаемые при детекции крупными точками и будем использовать их в качестве сегментации. полученную карту сегментации объединим с полученной первым способом. там достаточно хорошо выделяются крупные объекты - деревья и т.д. Также можно на этапе постпроцессинга трэчить объекты после детекции, например фильтром калмана.

Ноутбуки для детекции взяты отсюда:

https://www.kaggle.com/shonenkov/training-efficientdet

https://www.kaggle.com/shonenkov/inference-efficientdet

Для тренировки необходимо установить архивную версию пакета timm:

pip install --no-deps 'timm-0.1.26-py3-none-any.whl' > /dev/null



# Demo

python semantic_video.py --semantic_model Unet-Mobilenet.pt --detect_model effdet1_loss_0.6516653564969699_batch12_8000_state_dict.pt --colors_csv class_dict_seg.csv --inp_video video.mov

https://youtu.be/Sk1S_fkF6Js

