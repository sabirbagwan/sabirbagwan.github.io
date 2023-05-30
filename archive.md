---
layout: page
title: Blogs
---
<!--   <h3>{{ tag[0] }}</h3> -->

<!-- {% for tag in site.tags %}

  <h3>All Blogs</h3>
  <ul>
    {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.date | date: "%B %d, %Y" }} - {{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
 -->


<style>
.search-input {
/*   width: 400px; */
/*   width: 50% */
  width: 50vw;  
  height: 27px;
}
</style>


{% for tag in site.tags %}
  <!-- <h3>{{ tag[0] }}</h3> -->
<!--   <h3>All Blogs</h3> -->
<!--   <input type="text" id="search-input" placeholder="Search by title" onkeyup="searchBlogs()"> -->
    <a href="https://www.linkedin.com/in/sabirbagwan/recent-activity/articles/" target="_blank">Blogs on LinkedIn here</a> |
      <a href="https://medium.com/@sabirbagwan" target="_blank">Blogs On Medium here</a>
      <br>
  <input type="text" id="search-input" class="search-input" placeholder="Search by title or date" onkeyup="searchBlogs()">

  <ul id="blog-list">
    {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.date | date: "%B %d, %Y" }} - {{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}

<script>
function searchBlogs() {
  var input, filter, ul, li, a, i, txtValue;
  input = document.getElementById("search-input");
  filter = input.value.toUpperCase();
  ul = document.getElementById("blog-list");
  li = ul.getElementsByTagName("li");
  for (i = 0; i < li.length; i++) {
    a = li[i].getElementsByTagName("a")[0];
    txtValue = a.textContent || a.innerText;
    if (txtValue.toUpperCase().indexOf(filter) > -1) {
      li[i].style.display = "";
    } else {
      li[i].style.display = "none";
    }
  }
}
</script>
