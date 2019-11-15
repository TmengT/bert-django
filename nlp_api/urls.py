"""nlp_api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include, re_path
from django.conf.urls import url
from . import view


urlpatterns = [
    path('admin/', admin.site.urls),
    #上传文件
    re_path('upload/', view.upload_file,name='data'),
    re_path(r'^nlp$',view.Nlp_API.as_view(),name='nlp'),
    #测试html的变量赋值
    re_path(r'^hello$',view.hello_html,name='hello'),
    url(r'^$', view.Nlp_API.as_view()),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))

]
