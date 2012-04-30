import unittest

def main():
    tests = unittest.TestLoader().discover('.')
    result = unittest.TextTestRunner().run(tests)

if __name__ == "__main__":
    main()