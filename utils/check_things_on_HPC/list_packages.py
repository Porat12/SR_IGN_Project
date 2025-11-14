import pkg_resources

def list_all_packages():
    """
    Lists all installed packages and their versions using pkg_resources.
    """
    print("--- Listing All Installed Packages ---")
    
    # Get all distributions in the current working set (environment)
    installed_packages = sorted([
        f"- {d.key}=={d.version}" 
        for d in pkg_resources.working_set
    ])

    if installed_packages:
        for package in installed_packages:
            print(package)
    else:
        print("No packages found in the current environment.")

if __name__ == "__main__":
    list_all_packages()