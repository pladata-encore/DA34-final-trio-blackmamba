from django.shortcuts import render, get_object_or_404
from .models import Restaurant, Menu

def restaurant_list(request):
    restaurant_list = Restaurant.objects.order_by('id')
    context = {'restaurant_list': restaurant_list}
    return render(request, 'restaurant/restaurant_list.html', context)

def restaurant_detail(request, restaurant_id):
    restaurant = get_object_or_404(Restaurant, id=restaurant_id)
    menu = get_object_or_404(Menu, restaurant=restaurant)# 해당 레스토랑의 메뉴 가져오기
    # 이미지 파일 경로를 템플릿으로 전달하기 위해 context에 추가합니다.
    img_urls = []
    if restaurant.imgfile_1:
        img_urls.append(restaurant.imgfile_1.url)
    if restaurant.imgfile_2:
        img_urls.append(restaurant.imgfile_2.url)
    if restaurant.imgfile_3:
        img_urls.append(restaurant.imgfile_3.url)
    context = {'restaurant': restaurant,'menu': menu, 'img_urls': img_urls}
    return render(request, 'restaurant/restaurant_detail.html', context)


