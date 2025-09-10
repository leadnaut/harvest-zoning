
from zonings.models import Field, ZoningConfig
from zonings.zoning import make_zones

if __name__ == '__main__':
    f = Field(100, 100,
              [[i + j % 2 for j in range(100)] for i in range(100)], [], [])
    print(len(make_zones(f, ZoningConfig(30, 30))))
