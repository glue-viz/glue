import unittest

def main():
    tests = unittest.TestLoader().discover('.')
    result = unittest.TestResult()
    tests.run(result)
    print result

if __name__ == "__main__":
    main()