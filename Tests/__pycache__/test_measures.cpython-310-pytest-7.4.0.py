o
    ���d�  �                   @   sB   d dl Zd dlm  mZ d dlZd dlZd dlm	Z	 dd� Z
dS )�    N)�calculate_stratified_averagec                  C   s�  t �dg�} t �dg�}t| |�}d}||k}|sot�d|fd||f�dt�� v s.t�t�r3t�t�nddt�� v s?t�| �rDt�| �nddt�� v sPt�|�rUt�|�ndt�|�t�|�d� }dd	|i }t	t�
|���d  } }}t �td
��} t �td
d��}t| |�}d}||k}|s�t�d|fd||f�dt�� v s�t�t�r�t�t�nddt�� v s�t�| �r�t�| �nddt�� v s�t�|�r�t�|�ndt�|�t�|�d� }dd	|i }t	t�
|���d  } }}d S )N�   )�==)z9%(py4)s
{%(py4)s = %(py0)s(%(py1)s, %(py2)s)
} == %(py7)sr   �avg_asset_value�portion_of_investment)�py0�py1�py2�py4�py7zassert %(py9)s�py9�   �
   �P   )�numpy�arrayr   �
@pytest_ar�_call_reprcompare�@py_builtins�locals�_should_repr_global_name�	_saferepr�AssertionError�_format_explanation�range)r   r   �@py_assert3�@py_assert6�@py_assert5�@py_format8�@py_format10� r    �2/home/wasim/Desktop/EntroPy/Tests/test_measures.py�!test_calculate_stratified_average   s   ��r"   )�builtinsr   �_pytest.assertion.rewrite�	assertion�rewriter   �pytestr   �EntroPy.measuresr   r"   r    r    r    r!   �<module>   s   " 