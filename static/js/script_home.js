document.querySelectorAll("a[data-page]").forEach(function (link) {
    link.addEventListener("click", function (e) {
        e.preventDefault();
        const page = this.getAttribute('data-page');
        
        window.location.href = page;
        /*fetch(page)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error loading the page');
                }
                return response.text();
            })
            .then(data => {
                // แสดงเนื้อหาที่ได้ใน div#content-area
                document.getElementById('content-area').innerHTML = data;
            })
            .catch(error => {
                console.error('Error loading page:', error);
                document.getElementById('content-area').innerHTML = 'Error loading the page.';
            });*/
    });
});
/*history.pushState({ page: page }, "", `/${page}`);  // เปลี่ยน URL
        
        loadPage(page);  // โหลดหน้า
*/

document.addEventListener("click", function(event) {
    let link = event.target.closest('.menu-link');
    if (!link) return;
    
    event.preventDefault();
    const page = link.getAttribute('data-page');
    console.log("Loading page:", page);
    loadPage(page);
});

document.getElementById("go-home").addEventListener("click", function(e) {
    e.preventDefault();
    
    window.location.href = "/home";
});

function confirmLogout() {
    if (confirm("ต้องการที่จะออกจากระบบ ใช่หรือไม่?")) {
        document.getElementById("logout-Form").submit();
    } else {
        event.preventDefault();
    }
}

