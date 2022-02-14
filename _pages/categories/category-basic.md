---
title: "NLP"
layout: archive
permalink: categories/basic
author_profile: true
sidebar_main: true
typora-root-url: ../../
---

{% assign posts = site.categories.basic %}
{% for post in posts %} {% include archive-single2.html type=page.entries_layout %} {% endfor %}