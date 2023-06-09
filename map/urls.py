from . import views
from django.urls import path, re_path
from rest_framework.routers import DefaultRouter

urlpatterns = [
    path('upload', views.upload_img),
]

router = DefaultRouter()  # 可以处理视图的路由器
router.register(r'map', views.MapList)  # 向路由器中注册视图集

urlpatterns += router.urls  # 将路由器中的所以路由信息追到到django的路由列表中
