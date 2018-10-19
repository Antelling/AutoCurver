from curvefit import SummedCurver, WeightedCurver
from sklearn.utils.estimator_checks import check_estimator

check_estimator(SummedCurver(method="dogbox"))
check_estimator(WeightedCurver(method="dogbox"))

print("compatibility checks passed")