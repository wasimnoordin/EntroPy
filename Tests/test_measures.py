import pytest
import numpy

from EntroPy.measures import (
    calculate_stratified_average
    )

def test_calculate_stratified_average():
    avg_asset_value = numpy.array([1])
    portion_of_investment = numpy.array([1])
    assert calculate_stratified_average(avg_asset_value, portion_of_investment) == 1
    avg_asset_value = numpy.array(range(5))
    portion_of_investment = numpy.array(range(5, 10))
    assert calculate_stratified_average(avg_asset_value, portion_of_investment) == 80