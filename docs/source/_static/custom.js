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

  function lockSidebarShellsAtTop() {
    var shells = Array.prototype.slice.call(
      document.querySelectorAll('.wy-nav-side, .wy-side-scroll')
    );
    if (!shells.length) {
      return;
    }

    function resetShellScroll() {
      shells.forEach(function (shell) {
        if (shell.scrollTop !== 0) {
          shell.scrollTop = 0;
        }
        if (shell.scrollLeft !== 0) {
          shell.scrollLeft = 0;
        }
      });
    }

    shells.forEach(function (shell) {
      if (!shell.dataset.spaligndeScrollLock) {
        shell.dataset.spaligndeScrollLock = 'true';
        shell.addEventListener('scroll', resetShellScroll, { passive: true });
      }
    });

    if (!document.documentElement.dataset.spaligndeWindowScrollLock) {
      document.documentElement.dataset.spaligndeWindowScrollLock = 'true';
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
    lockSidebarShellsAtTop();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSidebar);
  } else {
    initializeSidebar();
  }
})();
