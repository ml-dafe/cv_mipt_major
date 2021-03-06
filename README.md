# "Компьютерное зрение"

**Описание:** Вы когда-нибудь задумывались, как роботы могут ориентироваться в пространстве и выполнять свои задачи, как поисковые системы могут индексировать миллиарды изображений и видео, как алгоритмы могут диагностировать медицинские изображения на предмет заболеваний, как автомобили с автоматическим управлением могут видеть и управлять автомобилем безопасно, или как Instagram создает фильтры?

В основе этих современных приложений ИИ лежат технологии компьютерного зрения, которые могут воспринимать, понимать и реконструировать сложный визуальный мир. Computer Vision – одна из самых быстрорастущих и захватывающих дисциплин искусственного интеллекта в современной академии и промышленности. Курс предназначен для того, чтобы познакомить студентов с постановками основных задач и основополагающих принципов на примерах частей реальных кейсов. В рамках программы будут рассмотрены классические подходы Computer Vison, знание которых является неотъемлемой частью и основой Computer Vision in Deep Learning.

**Содержание:** Классические методы обработки изображений, решение задач: классификации, распознавания и оценки параметров движения в видеопотоке.

**Что нужно знать и уметь:** Python (numpy, matplotlib), машинное обучение, линейная алгебра, математический анализ, статистика.

Курс основан на материалах [CS131](https://github.com/StanfordVL/CS131_release)

**Запись занятий:** [YouTube](https://youtube.com/playlist?list=PLQsqqFF3fDisnQkhNzBobVUEs1al10wr4) (rus)

## Перечень библиотек

1. Можно взять `requirements.txt`:

    ```pip install -r requirements.txt```
 
2. Вручную установить следующие пакеты:

| **Requirements** |
| :-- |
| `jupyter`        |
| `matplotlib`     |
| `cv2 (4.3)`      | 
| `skimage`        |
| `numpy`          |


## Программа курса:

00. [Вводное занятие:](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_00_introduction)
	- введение в Computer Vision - основные задачи и направления.
	
01. [Формирование изображений. Основные понятия:](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_01_images)
    - представление изображений в компьютере;
    - работа с цветом;
    - аффинные преобразования;
    - знакомство с библиотеками cv2, skimage;
    - [домашнее задание №1](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_01_images/homework).

02. [Введение в обработку сигналов:](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_02_signals)
	- частотная область, преобразование Фурье;
	- спектральный анализ;
	- свертки, фильтры;
	- кросс-корреляция;
	- [домашнее задание №2](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_02_signals/homework).
	
03. [Введение в обработку изображений:](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_03_image_processing)
	- гистограммы изображений;
	- цветовая коррекция;
	- пороговое выделение;
	- морфологические операции;
	- пирамиды изображений.

04. [Глобальные признаки изображений:](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_04_global_features)
	- выделение границ и контуров;
	- контураные признаки;
	- матрица смежности и текстурные признаки;
	- [домашнее задание №3](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_04_global_features/homework).
	
05. [Локальные признаки изображений:](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_05_local_features)
	- локализация особых точек (Harris, Shi-Tomasi);	
	- дескрипторы особых точек (SIFT).
    - [домашнее задание №4](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_05_local_features/homework).
	
06. [Сегментация на изображениях:](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_06_segmentation)
	- Задача сегментации;	
	- Иерархическая класетризация;
  	- Mean-Shift;
    - [домашнее задание №5](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_06_segmentation/homework).
	
07. [Параметрические модели:](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_07_parametric_models)
	- RANSAC;	
	- дескрипторы особых точек (SIFT);
  	- дескриптор изображений HOG;
	- пайплан задачи сшивки изображений (image stitching).
	
08. [Распознавание образов:](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_08_object_recognition)
	- постановка задачи, метрика качества;
	- простой детектор на основе HOG;
  	- Deformable parts model.

09. [Оптический видеопоток:](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_09_motion)
	- Оптический поток;
	- Lucas-Kanade method;
	- Horn-Schunk method;
	- Пирамиды для сильного движения;
	- Общий подход;
	- Применения.

10. [Задача сопровождения:](https://github.com/ml-dafe/cv_mipt_major/tree/main/week_10_tracking)
	- Feature Tracking;
	- Simple KLT tracker;
	- 2D transformations;
	- Iterative KLT tracker.
