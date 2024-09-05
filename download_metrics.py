import evaluate

if __name__ == "__main__":
    # cache metrics
    evaluate.load("accuracy")
    evaluate.load("mse")
    evaluate.load("f1")
    evaluate.load("seqeval")

