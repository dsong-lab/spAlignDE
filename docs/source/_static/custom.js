(function () {
  function moveLabLogoToSidebarBottom() {
    var sidebar = document.querySelector('.wy-side-scroll') || document.querySelector('.wy-nav-side');
    if (!sidebar) {
      return;
    }

    var oldVersionLogo = document.querySelector('.wy-side-nav-search .version');
    if (oldVersionLogo) {
      oldVersionLogo.remove();
    }

    var logoContainer = sidebar.querySelector('.lab-logo-container');
    if (!logoContainer) {
      logoContainer = document.createElement('div');
      logoContainer.className = 'lab-logo-container';
      sidebar.appendChild(logoContainer);
    }

    /*
     * Sphinx writes the correct relative documentation root on every page.
     * It is "./" at the site root, "../" one level down, and so on.  Using
     * this attribute keeps the logo URL valid on nested tutorial/notebook
     * pages; recent Sphinx versions no longer expose URL_ROOT through
     * DOCUMENTATION_OPTIONS.
     */
    var root = document.documentElement.getAttribute('data-content_root') || './';

    logoContainer.innerHTML = '<img class="lab-logo" src="' + root + '_static/lab-logo.png" alt="Lab logo">';
  }

  function removeLegacyHeNisselLinks() {
    var links = document.querySelectorAll('.wy-menu-vertical a');
    links.forEach(function (link) {
      var href = link.getAttribute('href') || '';
      var text = (link.textContent || '').replace(/\s+/g, ' ').trim().toLowerCase();
      if (
        href.indexOf('he_nissel_alignment.html') !== -1 ||
        text === 'h&e/nissel alignment' ||
        text === 'h&e / nissel alignment'
      ) {
        var item = link.closest('li');
        if (item) {
          item.remove();
        }
      }
    });
  }

  function lockSidebarShellAtTop() {
    var sidebar = document.querySelector('.wy-side-scroll');
    if (!sidebar) {
      return;
    }

    function resetShellScroll() {
      if (sidebar.scrollTop !== 0) {
        sidebar.scrollTop = 0;
      }
      if (sidebar.scrollLeft !== 0) {
        sidebar.scrollLeft = 0;
      }
    }

    if (!sidebar.dataset.spaligndeScrollLock) {
      sidebar.dataset.spaligndeScrollLock = 'true';
      sidebar.addEventListener('scroll', resetShellScroll, { passive: true });
      window.addEventListener('load', resetShellScroll);
      window.addEventListener('pageshow', resetShellScroll);
    }

    resetShellScroll();
    window.requestAnimationFrame(resetShellScroll);
    window.setTimeout(resetShellScroll, 0);
    window.setTimeout(resetShellScroll, 100);
  }

  function initializeSidebar() {
    moveLabLogoToSidebarBottom();
    removeLegacyHeNisselLinks();
    lockSidebarShellAtTop();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSidebar);
  } else {
    initializeSidebar();
  }
})();
