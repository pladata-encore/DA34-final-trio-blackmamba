from django.contrib import admin
from .models import Restaurant, Menu

class RestaurantAdmin(admin.ModelAdmin):
    search_fields =['title']
admin.site.register(Restaurant, RestaurantAdmin)
admin.site.register(Menu)

# Register your models here.
