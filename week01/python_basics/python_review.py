def calculate_statistics(numbers: list[float]) -> dict[str, float]:
    """
    Return the mean, minimum, maximum and range of a non-empty list.
    """
    if not numbers:
        raise ValueError("The list of numbers must not be empty.")

    mean = sum(numbers) / len(numbers)
    minimum = min(numbers)
    maximum = max(numbers)
    range_value = maximum - minimum

    return {
        "mean": mean,
        "min": minimum,
        "max": maximum,
        "range": range_value
    }

def normalize_scores(scores: list[float]) -> list[float]:
    """
    Apply min-max normalization.
    """    

    if not scores:
        raise ValueError("The list of scores must not be empty.")

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [0.0 for _ in scores]  # All scores are the same, return 0.0 for all

    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized_scores

numbers = [10, 20, 30, 40, 50]
stats = calculate_statistics(numbers)
normalized = normalize_scores(numbers)
print(stats)
print(normalized)

class ExperimentResult:
    def __init__(self, name: str, loss: float, accuracy: float):
        self.name = name
        self.loss = loss
        self.accuracy = accuracy

    def is_better_than(self, other: "ExperimentResult") -> bool:
        return self.accuracy > other.accuracy

a = ExperimentResult("Experiment A", loss=0.5, accuracy=0.8)
b = ExperimentResult("Experiment B", loss=0.4, accuracy=0.85)


assert calculate_statistics([1, 2, 3])["mean"] == 2
assert calculate_statistics([-1, -2, -3])["max"] == -1
assert normalize_scores([1, 2, 3]) == [0.0, 0.5, 1.0]
assert normalize_scores([5, 5, 5]) == [0.0, 0.0, 0.0]
assert a.is_better_than(b) == False
assert b.is_better_than(a) == True  
assert not b.is_better_than(a) == False  
