import pytest
from src import check_is_similar

class Test_TstTry(pytest):
    def test_2():
        assert(check_is_similar(1, 2), True)

if __name__ == '__main__':
    pytest.main()