from django.contrib import admin
from .models import ProductSearch, ProductResult

@admin.register(ProductSearch)
class ProductSearchAdmin(admin.ModelAdmin):
    list_display = ('id', 'search_date')
    readonly_fields = ('search_date',)

@admin.register(ProductResult)
class ProductResultAdmin(admin.ModelAdmin):
    list_display = ('title', 'price', 'source', 'similarity_score')
    list_filter = ('source',)
    search_fields = ('title',)